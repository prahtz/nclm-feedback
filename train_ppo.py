from valle.data.collation import get_text_token_collater
from lhotse.dataset import DynamicBucketingSampler
from torch.utils.data import DataLoader
from pathlib import Path
from models import wrap_and_save_valle_checkpoint, load_valle_with_value_head, VALLETokenizer
from reward_models import load_reward_model
from dataset import SpeechAlignmentDataset
import torch
from trl import PPOTrainer, PPOConfig, create_reference_model
from config import Parameters, get_default_parameters
import argparse
from tqdm import tqdm
from utils import get_gpu_usage_info, load_cuts, build_queries, AdamTFStyle, is_load_audios, repeater

from transformers import set_seed


def train_alignment(params: Parameters):
    model_path = wrap_and_save_valle_checkpoint(params, "prahtz/valle/")
    model = load_valle_with_value_head(model_path)
    model_ref = create_reference_model(model)
    valle_tokenizer = VALLETokenizer()
    train_cuts = load_cuts(params.data.manifest_dir, split="train")
    valid_cuts = load_cuts(params.data.manifest_dir, split="dev")
    text_token_collater = get_text_token_collater(params.data.text_tokens_file)
    reward_model = load_reward_model(params)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_audios = is_load_audios(params)
    train_dataset = SpeechAlignmentDataset(
        root_path=params.data.manifest_dir,
        split="train",
        text_token_collater=text_token_collater,
        load_audios=load_audios,
        bradley_terry_model=params.data.bradley_terry_model,
    )
    valid_dataset = SpeechAlignmentDataset(
        root_path=params.data.manifest_dir,
        split="dev",
        text_token_collater=text_token_collater,
        load_audios=load_audios,
        bradley_terry_model=False,
    )
    train_sampler = DynamicBucketingSampler(
        train_cuts,
        max_duration=params.data.max_duration,
        shuffle=True,
        buffer_size=params.data.buffer_size,
        shuffle_buffer_size=params.data.shuffle_buffer_size,
        quadratic_duration=10,
        num_cuts_for_bins_estimate=10000,
        drop_last=True,
    )
    valid_sampler = DynamicBucketingSampler(
        valid_cuts,
        max_duration=params.data.max_duration,
        shuffle=False,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=None,
        num_workers=params.sys.num_workers,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=None,
        num_workers=params.sys.num_workers,
        persistent_workers=True,
    )

    ppo_config = PPOConfig(
        batch_size=params.ppo.batch_size,
        mini_batch_size=params.ppo.mini_batch_size,
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_kwargs={"logging_dir": params.log.exp_dir},
        horizon=params.ppo.horizon,
        target=params.ppo.target,
        init_kl_coef=params.ppo.init_kl_coef,
        whiten_rewards=params.ppo.whiten_rewards,
        learning_rate=params.ppo.lr,
        cliprange_value=params.ppo.cliprange_value,
        vf_coef=params.ppo.vf_coef,
    )
    optimizer = AdamTFStyle(filter(lambda p: p.requires_grad, model.parameters()), lr=params.ppo.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=params.ppo.train_steps)
    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        model_ref,
        valle_tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    text_queries, audio_queries, responses, scores = [], [], [], []
    num_samples = 0
    num_ppo_steps = 0
    print("Starting training")
    for batch in tqdm(repeater(train_loader)):
        with torch.no_grad():
            prompts, prompts_lens = batch["prompts"].to(device), batch["prompts_lens"].to(device)
            text_tokens, text_tokens_lens = batch["text_tokens"].to(device), batch["text_tokens_lens"].to(device)
            time_invariant_codes = batch["prompts_time_invariant"].to(device)
            speaker_embeddings = batch["speaker_embeddings"].to(device)
            audios = [audio.to(device) for audio in batch["audios"]]
            texts = batch["text"]
            model.eval()
            response, response_lens = model.generate(
                x=text_tokens,
                x_lens=text_tokens_lens,
                y=prompts,
                y_lens=prompts_lens,
            )

            if torch.cuda.is_available():
                print(get_gpu_usage_info(), prompts_lens.detach() + response_lens.detach(), end=" ")

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

            if torch.cuda.is_available():
                print(get_gpu_usage_info())

            for i in range(prompts.shape[0]):
                responses.append(response[i, : response_lens[i], 0])
                text_queries.append(text_tokens[i, : text_tokens_lens[i]])
                audio_queries.append(prompts[i, -prompts_lens[i] :, 0])
                scores.append(reward[i])

            num_samples += prompts.shape[0]
        if num_samples >= params.ppo.batch_size:
            model.train()
            queries = build_queries(text_queries, audio_queries, params.ppo.batch_size, params.ppo.mini_batch_size)
            train_stats = ppo_trainer.step(
                queries[: params.ppo.batch_size], responses[: params.ppo.batch_size], scores[: params.ppo.batch_size]
            )
            print(train_stats["ppo/mean_scores"])
            ppo_trainer.log_stats(train_stats, {}, torch.stack(scores[: params.ppo.batch_size]))

            text_queries = text_queries[params.ppo.batch_size :]
            audio_queries = audio_queries[params.ppo.batch_size :]
            responses = responses[params.ppo.batch_size :]
            scores = scores[params.ppo.batch_size :]
            num_samples = num_samples - params.ppo.batch_size
            num_ppo_steps += 1

            if num_ppo_steps % 1000 == 0:
                model.save_pretrained(Path(params.log.exp_dir) / f"checkpoint-{num_ppo_steps}")

            if num_ppo_steps == params.ppo.train_steps:
                break
    model.save_pretrained(Path(params.log.exp_dir) / f"checkpoint-last")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    set_seed(1234)
    args = parser.parse_args()
    params = get_default_parameters()
    if args.cfg_path:
        params.merge_from_file(args.cfg_path)
    train_alignment(params)
