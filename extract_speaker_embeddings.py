import torch
from typing import List
from lhotse import load_manifest_lazy
from lhotse.dataset import UnsupervisedWaveformDataset
from lhotse.dataset import DynamicBucketingSampler
from torch.utils.data import DataLoader
import h5py
from pathlib import Path
from tqdm import tqdm
import nemo.collections.asr as nemo_asr
import torchaudio.functional as F


def compute_and_store_speaker_embeddings(root_path: str, manifest_names: List[str], num_workers: int = 0):
    root_path = Path(root_path)

    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
    speaker_model.eval()
    speaker_model.freeze()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speaker_model = speaker_model.to(device)

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
        with h5py.File(root_path / f"libritts_titanet_speaker_embeddings_{manifest_name}.h5", "w") as f_dataset, torch.no_grad():
            for batch in tqdm(loader):
                audio = batch["audio"]
                ids = batch["cuts"].ids
                audio = [F.resample(a[0].to(device), orig_freq=24000, new_freq=16000) for a in audio]
                audio_lens = torch.tensor([a.shape[0] for a in audio], device=device).long()
                audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)
                _, emb = speaker_model.forward(input_signal=audio, input_signal_length=audio_lens)

                for i, id in enumerate(ids):
                    f_dataset.create_dataset(id, data=emb[i].cpu().numpy())


if __name__ == "__main__":
    manifest_root_path = "valle/examples/libritts/data/tokenized/"
    # manifest_names = ["cuts_dev", "cuts_train", "cuts_test", "cuts_test_other"]
    manifest_names = ["cuts_test_other"]
    num_workers = 4
    compute_and_store_speaker_embeddings(root_path=manifest_root_path, manifest_names=manifest_names, num_workers=num_workers)
