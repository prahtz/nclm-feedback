import argparse
import torch

from tqdm import tqdm
from transformers import set_seed
from config import get_default_parameters, Parameters
from models import load_valle_with_value_head
from reward_models import (
    IncreaseLengthRewardModel,
    DecreaseLengthRewardModel,
    SpeakerSimilarityRewardModel,
    AudioQualityRewardModel,
    WERRewardModel,
)
from dataset import SpeechAlignmentDataset
from lhotse.dataset import DynamicBucketingSampler
from torch.utils.data import DataLoader
from valle.data.collation import get_text_token_collater
from utils import load_cuts
from collections import defaultdict
import pickle as pkl
from pathlib import Path
import os


def evaluate(params: Parameters, hf_ckpt_path: str, split: str):
    model = load_valle_with_value_head(hf_ckpt_path)
    reward_models = {
        "increase": IncreaseLengthRewardModel(),
        "decrease": DecreaseLengthRewardModel(),
        "speaker": SpeakerSimilarityRewardModel(params),
        "audio-quality": AudioQualityRewardModel(params),
        "wer": WERRewardModel(params),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    test_cuts = load_cuts(params.data.manifest_dir, split=split)
    text_token_collater = get_text_token_collater(params.data.text_tokens_file)
    test_dataset = SpeechAlignmentDataset(
        root_path=params.data.manifest_dir,
        split=split,
        text_token_collater=text_token_collater,
        load_audios=True,
        bradley_terry_model=False,
    )
    test_sampler = DynamicBucketingSampler(
        test_cuts,
        max_duration=params.data.max_duration,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=None,
        num_workers=params.sys.num_workers,
        persistent_workers=True,
    )
    model.eval()
    rewards = defaultdict(list)
    cuts_ids = []
    lengths = []
    print("Starting generation test...")
    for batch in tqdm(test_loader):
        with torch.no_grad():
            prompts, prompts_lens = batch["prompts"].to(device), batch["prompts_lens"].to(device)
            text_tokens, text_tokens_lens = batch["text_tokens"].to(device), batch["text_tokens_lens"].to(device)
            time_invariant_codes = batch["prompts_time_invariant"].to(device)
            speaker_embeddings = batch["speaker_embeddings"].to(device)
            audios = [audio.to(device) for audio in batch["audios"]]
            texts = batch["text"]
            ids = batch["utt_id"]

            response, response_lens = model.generate(x=text_tokens, x_lens=text_tokens_lens, y=prompts, y_lens=prompts_lens)

            for name, reward_model in reward_models.items():
                reward = reward_model(
                    texts=texts,
                    prompts=[prompts[i, -prompts_lens[i] :, 0] for i in range(prompts.shape[0])],
                    prompts_lens=prompts_lens,
                    responses=[response[i, : response_lens[i], 0] for i in range(response.shape[0])],
                    responses_lens=response_lens,
                    time_invariant_codes=time_invariant_codes,
                    speaker_embeddings=speaker_embeddings,
                    audios=audios,
                )
                rewards[name].append(reward.detach().cpu())
            lengths.append(response_lens.detach().cpu())
            cuts_ids.append(ids)

    cuts_ids = [id for ids in cuts_ids for id in ids]
    lengths = torch.cat(lengths)
    for name in rewards.keys():
        rewards[name] = torch.cat(rewards[name]).float()
    results = [cuts_ids, lengths, rewards]
    os.makedirs(Path(params.log.exp_dir), exist_ok=True)
    with open(Path(params.log.exp_dir) / f"evaluation_results_{split}.pkl", "wb") as f:
        pkl.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--hf-ckpt-path", type=str, default="prahtz/valle")
    set_seed(1234)
    args = parser.parse_args()
    params = get_default_parameters()
    if args.cfg_path:
        params.merge_from_file(args.cfg_path)
    evaluate(params, hf_ckpt_path=args.hf_ckpt_path, split=args.split)
