import torch
from typing import List
from lhotse import load_manifest_lazy
from lhotse.dataset import UnsupervisedWaveformDataset
from lhotse.dataset import DynamicBucketingSampler
from torch.utils.data import DataLoader
from valle.data.tokenizer import TiCodecAudioTokenizer
import h5py
from pathlib import Path
from tqdm import tqdm


def compute_and_store_time_invariant_codes(
    root_path: str, manifest_names: List[str], audio_tokenizer: TiCodecAudioTokenizer, num_workers: int = 0
):
    root_path = Path(root_path)

    for manifest_name in manifest_names:
        cuts = load_manifest_lazy(root_path / (manifest_name + ".jsonl.gz"))
        dataset = UnsupervisedWaveformDataset(collate=False)
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=400,
            shuffle=False,
            buffer_size=40000,
            shuffle_buffer_size=100000,
            quadratic_duration=10,
            num_cuts_for_bins_estimate=min(len(cuts), 10000),
            drop_last=False,
        )
        loader = DataLoader(dataset, batch_size=None, sampler=sampler, shuffle=False, drop_last=False, num_workers=num_workers)

        print(f"Processing {manifest_name}")
        with h5py.File(root_path / f"libritts_ticodec_time_invariant_{manifest_name}.h5", "w") as f_dataset, torch.no_grad():
            for batch in tqdm(loader):
                audio = batch["audio"]
                ids = batch["cuts"].ids
                audio_lens = torch.tensor([a.shape[1] for a in audio]).long()
                prompts_lens = torch.min(0.25 * (audio_lens // 320), torch.ones(audio_lens.shape[0]) * 225).long()
                audio = [audio[i][0, : prompts_lens[i] * 320] for i in range(len(audio))]
                audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)
                _, global_codes = audio_tokenizer.encode(audio)
                for i, id in enumerate(ids):
                    f_dataset.create_dataset(id, data=global_codes[i].cpu().numpy())


if __name__ == "__main__":
    manifest_root_path = "valle/examples/libritts/data/tokenized/"
    # manifest_names = ["cuts_dev", "cuts_train", "cuts_test", "cuts_test_other"]
    manifest_names = ["cuts_test_other"]
    config_path = "TiCodec/egs/TiCodec-24k-320d/config_24k_320d_conv_1g1r_8g3k1s.json"
    ckpt_path = "TiCodec/checkpoints/1codebook/g_00300000"
    audio_tokenizer = TiCodecAudioTokenizer(config_path=config_path, ckpt_path=ckpt_path)
    num_workers = 4
    compute_and_store_time_invariant_codes(
        root_path=manifest_root_path, manifest_names=manifest_names, audio_tokenizer=audio_tokenizer, num_workers=num_workers
    )
