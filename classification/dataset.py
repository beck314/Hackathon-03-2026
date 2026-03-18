import os
import json
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import english_cnameEncoder
import pickle


# ── Configuration ─────────────────────────────────────────────────────────────

class Config:
    # Paths
    FLAC_DIR      = "archive/songs/songs"          # directory containing .flac files
    CSV_PATH      = "archive/birdsong_metadata.csv"     # must have columns: file_id, english_cname
    CACHE_DIR     = "data/cache"          # pre-computed spectrograms saved here
    MODEL_DIR     = "models"

    # Audio
    SAMPLE_RATE   = 22050
    CLIP_DURATION = 5.0                   # seconds — all clips padded/trimmed to this
    N_FFT         = 2048
    HOP_LENGTH    = 512
    N_MELS        = 128                   # mel spectrogram height
    FMIN          = 500                   # birds rarely sing below 500 Hz
    FMAX          = 15000                 # upper frequency cutoff

    # Model
    N_EPOCHS      = 40
    BATCH_SIZE    = 32
    LR            = 3e-4
    VAL_SPLIT     = 0.15
    TEST_SPLIT    = 0.15
    SEED          = 42


# ── Audio utilities ────────────────────────────────────────────────────────────

def load_clip(path: str, cfg: Config) -> np.ndarray:
    """Load, resample, and pad/trim to a fixed-length waveform."""
    y, _ = librosa.load(path, sr=cfg.SAMPLE_RATE, mono=True,
                        duration=cfg.CLIP_DURATION)
    target = int(cfg.CLIP_DURATION * cfg.SAMPLE_RATE)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y


def augment_waveform(y: np.ndarray, sr: int) -> np.ndarray:
    """Light data augmentation on the raw waveform."""
    # Time stretch (±10%)
    rate = np.random.uniform(0.9, 1.1)
    y = librosa.effects.time_stretch(y, rate=rate)

    # Pitch shift (±2 semitones)
    steps = np.random.uniform(-2, 2)
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

    # Add a little Gaussian noise
    y = y + np.random.normal(0, 0.005, len(y)).astype(np.float32)

    return y


def to_log_mel(y: np.ndarray, cfg: Config) -> np.ndarray:
    """Convert waveform → log-mel spectrogram, shape (1, N_MELS, T)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.SAMPLE_RATE,
        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return log_mel[np.newaxis, ...]          # add channel dim → (1, 128, T)


def build_cache(cfg: Config, english_cname_encoder: english_cnameEncoder,
                df: pd.DataFrame) -> None:
    """Pre-compute and cache all spectrograms to disk (run once)."""
    Path(cfg.CACHE_DIR).mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        stem = Path(row["file_id"]).stem
        out  = Path(cfg.CACHE_DIR) / f"{stem}.npy"
        if out.exists():
            continue
        path = str(Path(cfg.FLAC_DIR) / row["file_id"])
        try:
            y       = load_clip(path, cfg)
            log_mel = to_log_mel(y, cfg)
            np.save(out, log_mel)
        except Exception as e:
            print(f"  ⚠ Skipping {row['file_id']}: {e}")


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class BirdsongDataset(Dataset):
    def __init__(self, df: pd.DataFrame, english_cname_encoder: english_cnameEncoder,
                 cfg: Config, augment: bool = False):
        self.df            = df.reset_index(drop=True)
        self.english_cname_encoder = english_cname_encoder
        self.cfg           = cfg
        self.augment       = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        stem = Path(row["file_id"]).stem
        cache_path = Path(self.cfg.CACHE_DIR) / f"{stem}.npy"

        if cache_path.exists():
            log_mel = np.load(cache_path)
            if self.augment:
                # Re-compute from waveform so augmentation is applied
                path = str(Path(self.cfg.FLAC_DIR) / row["file_id"])
                y       = load_clip(path, self.cfg)
                y       = augment_waveform(y, self.cfg.SAMPLE_RATE)
                log_mel = to_log_mel(y, self.cfg)
        else:
            path    = str(Path(self.cfg.FLAC_DIR) / row["file_id"])
            y       = load_clip(path, self.cfg)
            if self.augment:
                y = augment_waveform(y, self.cfg.SAMPLE_RATE)
            log_mel = to_log_mel(y, self.cfg)

        # SpecAugment: randomly mask frequency and time bands
        if self.augment:
            log_mel = spec_augment(log_mel)

        english_cname = self.english_cname_encoder.transform([row["english_cname"]])[0]
        return torch.tensor(log_mel), torch.tensor(english_cname, dtype=torch.long)


def spec_augment(spec: np.ndarray,
                 freq_mask_max: int = 20,
                 time_mask_max: int = 30,
                 n_masks: int = 2) -> np.ndarray:
    """SpecAugment: zero out random frequency and time bands."""
    spec = spec.copy()
    _, F, T = spec.shape
    for _ in range(n_masks):
        f = np.random.randint(0, freq_mask_max)
        f0 = np.random.randint(0, max(1, F - f))
        spec[:, f0:f0 + f, :] = 0

        t = np.random.randint(0, time_mask_max)
        t0 = np.random.randint(0, max(1, T - t))
        spec[:, :, t0:t0 + t] = 0
    return spec