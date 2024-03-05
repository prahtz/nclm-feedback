import torch
from transformers import Wav2Vec2Processor, HubertForCTC
from valle.data.tokenizer import TiCodecAudioTokenizer
from config import Parameters
from torch.nn.utils.rnn import pad_sequence
import librosa
import regex
from torchmetrics.functional import word_error_rate
import torchaudio.functional as F
import nemo.collections.asr as nemo_asr
from torchaudio.pipelines import SQUIM_SUBJECTIVE


class RewardModel:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class ConstantRewardModel(RewardModel):
    def forward(self, responses_lens, **kwargs):
        return torch.ones(responses_lens.shape, device=responses_lens.device)


class LinearDecayRewardModel(RewardModel):
    def __init__(self, params: Parameters):
        self.batch_size = params.ppo.batch_size
        self.reward = torch.tensor(1.0)
        self.std = torch.tensor(0.1)
        self.step = self.reward / params.ppo.train_steps
        self.std_step = self.std / params.ppo.train_steps
        self.tot_calls = 0

    def forward(self, responses_lens, **kwargs):
        current_reward = torch.normal(self.reward, 0.05)
        current_std = max(torch.tensor(1e-8), torch.normal(self.std, 0.05))
        reward_tensor = torch.tensor(
            [torch.normal(current_reward, current_std) for _ in range(responses_lens.shape[0])], device=responses_lens.device
        )
        self.tot_calls += responses_lens.shape[0]
        if self.tot_calls >= self.batch_size:
            self.tot_calls = self.tot_calls - self.batch_size
            self.reward = self.reward - self.step
            self.std = self.std + self.std_step
        return reward_tensor


class IncreaseLengthRewardModel(RewardModel):
    def forward(self, prompts_lens, responses_lens, **kwargs):
        reward_tensor = torch.empty(responses_lens.shape, dtype=torch.float32, device=responses_lens.device)
        for i in range(responses_lens.shape[0]):
            reward_tensor[i] = responses_lens[i] / (prompts_lens[i] * 6.666)
        return reward_tensor


class DecreaseLengthRewardModel(RewardModel):
    min_length = 50

    def forward(self, prompts_lens, responses_lens, **kwargs):
        reward_tensor = torch.empty(responses_lens.shape, dtype=torch.float32, device=responses_lens.device)
        for i in range(responses_lens.shape[0]):
            max_length = prompts_lens[i] * 6.666
            reward_tensor[i] = (max_length - responses_lens[i]) / (max_length - self.min_length)
            reward_tensor[i] = 0.0 if reward_tensor[i] > 1.0 else reward_tensor[i]
        return reward_tensor


class WERRewardModel(RewardModel):
    def __init__(self, params: Parameters):
        self.audio_tokenizer = TiCodecAudioTokenizer(config_path=params.model.ticodec_config_path, ckpt_path=params.model.ticodec_ckpt_path)
        self.asr_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.asr_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        if torch.cuda.is_available():
            self.asr_model = self.asr_model.to("cuda")
        self.asr_model.eval()

        self.all_but_chars = regex.compile("[^A-Z' ]")
        self.multi_whitespaces = regex.compile(" +")

    def __call__(self, texts, prompts, prompts_lens, responses, responses_lens, time_invariant_codes, **kwargs):
        ids_nonzero = None
        bsz = len(prompts)
        with torch.no_grad():
            total_lens = prompts_lens + responses_lens
            # filter empty responses
            if (responses_lens == 1).all():
                return torch.zeros(len(prompts), device=responses_lens.device, dtype=torch.float32)
            if (responses_lens == 1).any():
                ids_nonzero = torch.nonzero(responses_lens != 1).squeeze(-1)
                responses, responses_lens = [responses[i.item()] for i in ids_nonzero], responses_lens[ids_nonzero]
                prompts = [prompts[i.item()] for i in ids_nonzero]
                time_invariant_codes = time_invariant_codes[ids_nonzero]
                texts = [texts[i.item()] for i in ids_nonzero]
                total_lens = total_lens[ids_nonzero]
            codes = [torch.cat([prompts[i], responses[i][:-1]]) for i in range(len(prompts))]
            codes = pad_sequence(codes, batch_first=True, padding_value=0).unsqueeze(-1)
            wavs = self.audio_tokenizer.decode(codes=codes, global_codes=time_invariant_codes).squeeze(1)
            wavs_lens = (total_lens - 1) * 320
            wavs = [wavs[i, : wavs_lens[i]].cpu().numpy() for i in range(wavs.shape[0])]
            wavs = [librosa.resample(wav, orig_sr=24000, target_sr=16000) for wav in wavs]

            out = self.asr_processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
            wavs, attention_mask = out.input_values.to(prompts[0].device), out.attention_mask.to(prompts[0].device)

            logits = self.asr_model(wavs, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_texts = self.asr_processor.batch_decode(pred_ids)
            texts = [regex.sub(self.all_but_chars, " ", text.upper().strip("'")) for text in texts]
            texts = [regex.sub(self.multi_whitespaces, " ", text).strip() for text in texts]
            wers = [word_error_rate(preds=[pred_texts[i]], target=[texts[i]]).item() for i in range(len(texts))]
            wers = 1 - torch.tensor(wers, device=pred_ids.device)
            if ids_nonzero is not None:
                temp = wers
                wers = torch.zeros(bsz, device=wers.device, dtype=torch.float32)
                wers[ids_nonzero] = temp
            return wers


class SpeakerSimilarityRewardModel(RewardModel):
    def __init__(self, params: Parameters):
        self.audio_tokenizer = TiCodecAudioTokenizer(config_path=params.model.ticodec_config_path, ckpt_path=params.model.ticodec_ckpt_path)
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
        if torch.cuda.is_available():
            self.speaker_model = self.speaker_model.to("cuda")
        self.speaker_model.eval()
        self.speaker_model.freeze()

    def forward(self, responses, responses_lens, time_invariant_codes, speaker_embeddings, **kwargs):
        ids_nonzero = None
        bsz = len(responses)
        with torch.no_grad():
            # filter empty responses
            if (responses_lens == 1).all():
                return torch.zeros(bsz, device=responses_lens.device, dtype=torch.float32)
            if (responses_lens == 1).any():
                ids_nonzero = torch.nonzero(responses_lens != 1).squeeze(-1)
                responses, responses_lens = [responses[i.item()] for i in ids_nonzero], responses_lens[ids_nonzero]
                time_invariant_codes = time_invariant_codes[ids_nonzero]
                speaker_embeddings = speaker_embeddings[ids_nonzero]
            codes = [responses[i][:-1] for i in range(len(responses))]
            codes = pad_sequence(codes, batch_first=True, padding_value=0).unsqueeze(-1)
            wavs = self.audio_tokenizer.decode(codes=codes, global_codes=time_invariant_codes).squeeze(1)
            wavs_lens = (responses_lens - 1) * 320
            wavs = [wavs[i, : wavs_lens[i]] for i in range(wavs.shape[0])]
            wavs = [F.resample(wav, orig_freq=24000, new_freq=16000) for wav in wavs]
            wavs_lens = torch.tensor([wavs[i].shape[0] for i in range(len(wavs))], device=wavs[0].device, dtype=torch.int32)
            wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
            _, emb = self.speaker_model.forward(input_signal=wavs, input_signal_length=wavs_lens)

            # Length Normalize
            speaker_embeddings = speaker_embeddings / torch.linalg.norm(speaker_embeddings, dim=1).unsqueeze(-1)
            emb = emb / torch.linalg.norm(emb, dim=1).unsqueeze(-1)
            # Score
            similarity_scores = torch.bmm(speaker_embeddings.unsqueeze(1), emb.unsqueeze(-1))[:, 0, 0]
            similarity_scores = (similarity_scores + 1) / 2

            if ids_nonzero is not None:
                temp = similarity_scores
                similarity_scores = torch.zeros(bsz, device=similarity_scores.device, dtype=torch.float32)
                similarity_scores[ids_nonzero] = temp
            return similarity_scores


class AudioQualityRewardModel(RewardModel):
    def __init__(self, params: Parameters):
        self.audio_tokenizer = TiCodecAudioTokenizer(config_path=params.model.ticodec_config_path, ckpt_path=params.model.ticodec_ckpt_path)
        self.mos_predictor_model = SQUIM_SUBJECTIVE.get_model()
        if torch.cuda.is_available():
            self.mos_predictor_model = self.mos_predictor_model.to("cuda")

    def forward(self, prompts_lens, responses, responses_lens, time_invariant_codes, audios, **kwargs):
        ids_nonzero = None
        bsz = len(responses)
        with torch.no_grad():
            # filter empty responses
            if (responses_lens <= 2).all():
                return torch.zeros(bsz, device=responses_lens.device, dtype=torch.float32)
            if (responses_lens <= 2).any():
                ids_nonzero = torch.nonzero(responses_lens > 2).squeeze(-1)
                prompts_lens = prompts_lens[ids_nonzero]
                responses, responses_lens = [responses[i.item()] for i in ids_nonzero], responses_lens[ids_nonzero]
                time_invariant_codes = time_invariant_codes[ids_nonzero]
                audios = [audios[i.item()] for i in ids_nonzero]
            prompts_lens = prompts_lens * 320
            audios = [audios[i][0, prompts_lens[i] :] for i in range(len(audios))]
            audios = [F.resample(audios[i], orig_freq=24000, new_freq=16000) for i in range(len(audios))]

            codes = [responses[i][:-1] for i in range(len(responses))]  # removing EOS token
            codes = pad_sequence(codes, batch_first=True, padding_value=0).unsqueeze(-1)
            wavs = self.audio_tokenizer.decode(codes=codes, global_codes=time_invariant_codes).squeeze(1)
            wavs_lens = (responses_lens - 1) * 320
            wavs = [wavs[i, : wavs_lens[i]] for i in range(wavs.shape[0])]
            wavs = [F.resample(wav, orig_freq=24000, new_freq=16000) for wav in wavs]

            mos = []
            for a, wav in zip(audios, wavs):
                mos.append(self.mos_predictor_model(wav.unsqueeze(0), a.unsqueeze(0)))
            mos = torch.cat(mos) / 5.0

            if ids_nonzero is not None:
                temp = mos
                mos = torch.zeros(bsz, device=mos.device, dtype=torch.float32)
                mos[ids_nonzero] = temp
            return mos


def load_reward_model(params: Parameters) -> RewardModel:
    if params.reward.model_name == "decrease-length":
        return DecreaseLengthRewardModel()
    elif params.reward.model_name == "increase-length":
        return IncreaseLengthRewardModel()
    elif params.reward.model_name == "constant":
        return ConstantRewardModel()
    elif params.reward.model_name == "linear-decay":
        return LinearDecayRewardModel(params)
    elif params.reward.model_name == "wer":
        return WERRewardModel(params)
    elif params.reward.model_name == "speaker":
        return SpeakerSimilarityRewardModel(params)
    elif params.reward.model_name == "audio-quality":
        return AudioQualityRewardModel(params)
    raise RuntimeError(f'Unsupported reward model name: "{params.reward.model_name}"')
