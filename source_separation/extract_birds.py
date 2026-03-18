"""
extract_birds.py
----------------
Takes noisy .wav files, extracts only bird sounds, saves as .flac.

Install:
    pip install demucs soundfile numpy

Usage:
    # Single file
    python extract_birds.py recording.wav

    # Whole folder
    python extract_birds.py ./field_recordings/

    # Specify output folder
    python extract_birds.py ./field_recordings/ --output ./birds_only/
"""

import argparse
import sys
import torch

from pathlib import Path

from demucs.pretrained import get_model
from demucs.apply import apply_model

import numpy as np
import soundfile as sf


SAMPLE_RATE = 44_100

# Bird songs live mostly between 1 kHz and 10 kHz.
BIRD_FREQ_LOW  = 1_000   # Hz
BIRD_FREQ_HIGH = 10_000  # Hz

def load_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)          # stereo → mono
    return audio.astype(np.float32), sr


def save_flac(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, format="FLAC")


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Basic linear interpolation fallback
        n_out = int(len(audio) * target_sr / orig_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

def spectral_gate(audio: np.ndarray, sr: int) -> np.ndarray:
    fft    = np.fft.rfft(audio)
    freqs  = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    mask   = (freqs >= BIRD_FREQ_LOW) & (freqs <= BIRD_FREQ_HIGH)
    return np.fft.irfft(fft * mask, n=len(audio)).astype(np.float32)

def wiener_denoise(audio: np.ndarray, frame: int = 1024) -> np.ndarray:
    if len(audio) < frame * 2:
        return audio

    padded  = np.pad(audio, (0, frame - len(audio) % frame))
    frames  = padded.reshape(-1, frame)
    spec    = np.fft.rfft(frames, axis=1)
    mag     = np.abs(spec)

    # Estimate noise floor from quietest 10 % of frames
    energy      = mag.mean(axis=1)
    n_noise     = max(1, int(0.10 * len(energy)))
    noise_psd   = mag[np.argsort(energy)[:n_noise]].mean(axis=0)

    gain        = np.maximum(1.0 - noise_psd / (mag + 1e-9), 0.1)
    filtered    = np.fft.irfft(spec * gain, n=frame, axis=1).astype(np.float32)
    return filtered.flatten()[: len(audio)]

def demucs_separate(audio: np.ndarray, sr: int, model_name: str = "htdemucs") -> np.ndarray:
    """
    Runs Demucs and returns the 'vocals' stem, which captures high-frequency
    tonal content closest to bird song.  Falls back to spectral gate only if
    Demucs / torch are not installed.
    """
    try:
        model = get_model(model_name)
        model.eval()

        wav_t = torch.tensor(audio[None, :], dtype=torch.float32)
        wav_t = wav_t.repeat(1, 2, 1)  # mono → stereo
        with torch.no_grad():
            sources = apply_model(model, wav_t, device="cpu")   # [1, stems, ch, time]

        stem_names = list(model.sources)
        idx = stem_names.index("vocals") if "vocals" in stem_names else 0
        stem = sources[0, idx].mean(0).cpu().numpy().astype(np.float32)
        print("    [demucs] separation complete")
        return stem

    except ImportError:
        print("    [fallback] demucs not found — using spectral gate only")
        print("               install with:  pip install demucs")
        return audio   # passthrough; spectral_gate + wiener still run below

def extract_birds(input_path: Path, output_path: Path, model: str) -> None:
    print(f"  loading  : {input_path.name}")
    audio, sr = load_wav(input_path)

    if sr != SAMPLE_RATE:
        print(f"    resampling {sr} Hz → {SAMPLE_RATE} Hz")
        audio = resample(audio, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Normalise input level
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.98

    print(f"    separating with demucs ({model}) …")
    bird = demucs_separate(audio, sr, model)

    print("    applying spectral gate  [1 kHz – 10 kHz] …")
    bird = spectral_gate(bird, sr)

    print("    denoising …")
    bird = wiener_denoise(bird)

    # Final normalise
    peak = np.max(np.abs(bird))
    if peak > 0:
        bird = bird / peak * 0.95

    save_flac(output_path, bird, sr)
    print(f"  saved    : {output_path}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract bird sounds from noisy .wav files → .flac"
    )
    p.add_argument("input", type=Path,
                   help="A .wav file or a directory of .wav files")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output file or directory (default: ./birds_only/)")
    p.add_argument("--model", default="htdemucs",
                   choices=["htdemucs", "htdemucs_ft", "mdx_extra"],
                   help="Demucs model to use (default: htdemucs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Collect input files
    if args.input.is_dir():
        wav_files = sorted(args.input.glob("*.wav"))
        if not wav_files:
            sys.exit(f"No .wav files found in {args.input}")
        out_dir = args.output or Path("birds_only")
        pairs = [(f, out_dir / f.with_suffix(".flac").name) for f in wav_files]
    elif args.input.is_file():
        out = args.output or args.input.with_suffix(".flac")
        pairs = [(args.input, out)]
    else:
        sys.exit(f"Input not found: {args.input}")

    print(f"\nBird extractor — {len(pairs)} file(s)\n")
    for i, (src, dst) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}]")
        try:
            extract_birds(src, dst, args.model)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()