import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import whisper


class Config:
    FLAC_DIR      = "archive/songs/songs"
    CSV_PATH      = "archive/birdsong_metadata.csv"
    CACHE_DIR     = "data/cache"
    MODEL_DIR     = "models"

    # Must match Whisper's expected input
    SAMPLE_RATE   = 16_000
    CLIP_DURATION = 5.0
    N_MELS        = 80

    N_EPOCHS      = 20
    BATCH_SIZE    = 32
    LR            = 3e-4
    VAL_SPLIT     = 0.1
    TEST_SPLIT    = 0.35
    SEED          = 42


def load_clip(path: str, cfg: Config) -> torch.Tensor:
    """Load and return a fixed-length mono waveform at 16kHz."""
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != cfg.SAMPLE_RATE:
        waveform = T.Resample(sr, cfg.SAMPLE_RATE)(waveform)

    target = int(cfg.CLIP_DURATION * cfg.SAMPLE_RATE)
    if waveform.shape[1] < target:
        waveform = torch.nn.functional.pad(waveform, (0, target - waveform.shape[1]))
    else:
        waveform = waveform[:, :target]

    return waveform.squeeze(0)  # (samples,)


def to_log_mel(waveform: torch.Tensor, cfg: Config) -> torch.Tensor:
    """
    Convert waveform → Whisper-compatible log-mel spectrogram (80, T).
    Uses whisper.log_mel_spectrogram to guarantee encoder compatibility.
    """
    # Whisper's function expects exactly 30s of audio at 16kHz
    chunk = whisper.audio.SAMPLE_RATE * whisper.audio.CHUNK_LENGTH
    if len(waveform) < chunk:
        waveform = torch.nn.functional.pad(waveform, (0, chunk - len(waveform)))
    else:
        waveform = waveform[:chunk]

    mel = whisper.log_mel_spectrogram(waveform)  # (80, 3000)

    # Slice to our clip duration (3000 frames = 30s, so 5s = 500 frames)
    t_frames = int(cfg.CLIP_DURATION / whisper.audio.CHUNK_LENGTH * mel.shape[1])
    return mel[:, :t_frames]  # (80, T)


def augment(waveform: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Speed perturb + light noise."""
    speed    = float(np.random.uniform(0.9, 1.1))
    waveform = T.Resample(cfg.SAMPLE_RATE, int(cfg.SAMPLE_RATE * speed))(waveform.unsqueeze(0)).squeeze(0)

    target = int(cfg.CLIP_DURATION * cfg.SAMPLE_RATE)
    if len(waveform) < target:
        waveform = torch.nn.functional.pad(waveform, (0, target - len(waveform)))
    else:
        waveform = waveform[:target]

    return waveform + torch.randn_like(waveform) * 0.005


def build_cache(cfg: Config, df: pd.DataFrame) -> None:
    Path(cfg.CACHE_DIR).mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        stem = Path(("xc"+str(row["file_id"]))).stem
        out  = Path(cfg.CACHE_DIR) / f"{stem}.npy"
        if out.exists():
            continue
        try:
            wav = load_clip(str(Path(cfg.FLAC_DIR) / ("xc"+str(row["file_id"]))), cfg)
            np.save(out, to_log_mel(wav, cfg).numpy())
        except Exception as e:
            print(f"  ⚠ Skipping {("xc"+str(row['file_id']))}: {e}")


class BirdsongDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder,
                 cfg: Config, augment_data: bool = False):
        self.df           = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.cfg          = cfg
        self.augment_data = augment_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        stem       = Path(("xc"+str(row["file_id"]))).stem
        cache_path = Path(self.cfg.CACHE_DIR) / f"{stem}.npy"

        if cache_path.exists() and not self.augment_data:
            mel = torch.tensor(np.load(cache_path))
        else:
            wav = load_clip(str(Path(self.cfg.FLAC_DIR) / ("xc"+str(row["file_id"]))), self.cfg)
            if self.augment_data:
                wav = augment(wav, self.cfg)
            mel = to_log_mel(wav, self.cfg)

        label = self.label_encoder.transform([row["english_cname"]])[0]
        return mel, torch.tensor(label, dtype=torch.long)