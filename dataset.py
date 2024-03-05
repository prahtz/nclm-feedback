from typing import Dict
import torch
from torch.utils.data import Dataset
from lhotse import CutSet
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_audio

from valle.data.collation import TextTokenCollater
import torch.nn.functional as F
import h5py
from pathlib import Path


class SpeechAlignmentDataset(Dataset):
    """
    The PyTorch Dataset for the speech synthesis(e.g. TTS) task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'audio_lens': (B, ) int tensor
            'text': str
            'audio_features': (B x NumFrames x NumFeatures) float tensor
            'audio_features_lens': (B, ) int tensor
            'text_tokens': (B x NumTextTokens) long tensor
            'text_tokens_lens': (B, ) int tensor
        }
    """

    def __init__(
        self,
        root_path: str,
        split: str,
        text_token_collater: TextTokenCollater,
        load_audios: bool = False,
        bradley_terry_model: bool = False,
    ) -> None:
        super().__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.text_token_collater = text_token_collater
        self.load_audios = load_audios
        self.text_pad_token = self.text_token_collater.token2idx[self.text_token_collater.pad_symbol]
        self.f_time_invariant_dataset = h5py.File(self.root_path / f"libritts_ticodec_time_invariant_cuts_{self.split}.h5", "r")
        self.f_speaker_embeddings_dataset = h5py.File(self.root_path / f"libritts_titanet_speaker_embeddings_cuts_{self.split}.h5", "r")
        self.bradley_terry_model = bradley_terry_model

    def collate_features(self, features, pad_token: int = 0, pad_direction: str = "right"):
        feature_lens = torch.tensor([feat.shape[0] for feat in features]).long()
        max_len = feature_lens.max().item()
        if pad_direction == "left":
            features = [F.pad(feat, (0, 0, max_len - feat.shape[0], 0), value=pad_token) for feat in features]
        elif pad_direction == "right":
            features = [F.pad(feat, (0, 0, 0, max_len - feat.shape[0]), value=pad_token) for feat in features]
        return torch.stack(features), feature_lens

    def load_time_invariant_features(self, cuts: CutSet):
        return torch.cat([torch.from_numpy(self.f_time_invariant_dataset[cut.id][:]) for cut in cuts]).unsqueeze(1)

    def load_speaker_embeddings(self, cuts: CutSet):
        return torch.stack([torch.from_numpy(self.f_speaker_embeddings_dataset[cut.id][:]) for cut in cuts])

    def prepare_prompts_and_targets(self, cuts: CutSet):
        features = [torch.from_numpy(cut.load_features()) for cut in cuts]
        feature_lens = torch.tensor([feat.shape[0] for feat in features]).long()
        prompts_lens = torch.tensor([min((0.25 * feature_lens[i]).int().item(), 255) for i in range(len(features))]).long()
        prompts = [features[i][: prompts_lens[i]] for i in range(len(features))]
        targets = [features[i][prompts_lens[i] :] for i in range(len(features))]
        prompts, prompts_lens = self.collate_features(prompts, pad_token=self.text_pad_token, pad_direction="left")
        targets, targets_lens = self.collate_features(targets, pad_token=0, pad_direction="right")
        return prompts, prompts_lens, targets, targets_lens

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        if self.bradley_terry_model:
            cuts = [[cut, cut] for cut in cuts]
            cuts = [cut for pair in cuts for cut in pair]
            cuts = {i: cut for i, cut in enumerate(cuts)}
            cuts = CutSet(cuts)
        audios = []
        if self.load_audios:
            audios = [torch.from_numpy(cut.load_audio()) for cut in cuts]

        prompts, prompts_lens, targets, targets_lens = self.prepare_prompts_and_targets(cuts)
        prompts_time_invariant = self.load_time_invariant_features(cuts)
        speaker_embeddings = self.load_speaker_embeddings(cuts)
        text_tokens, text_tokens_lens = self.text_token_collater([cut.supervisions[0].custom["tokens"]["text"] for cut in cuts])

        return {
            "utt_id": [cut.id for cut in cuts],
            "text": [cut.supervisions[0].text for cut in cuts],
            "audios": audios,
            "prompts": prompts,
            "prompts_lens": prompts_lens,
            "targets": targets,
            "targets_lens": targets_lens,
            "prompts_time_invariant": prompts_time_invariant,
            "text_tokens": text_tokens,
            "text_tokens_lens": text_tokens_lens,
            "speaker_embeddings": speaker_embeddings,
        }
