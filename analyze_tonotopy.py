"""
Analyze and visualize tonotopic organization in trained conv1 filters

Usage examples:
    analyze both models
    python analyze_tonotopy.py \
        --neuro-ckpt path/to/best_neuro.pt \
        --baseline-ckpt path/to/best_baseline.pt \
        --outdir ./analysis --n-freqs 48

    analyze single model
    python analyze_tonotopy.py --neuro-ckpt path/to/best_neuro.pt --outdir ./analysis

Dependencies:
    - torch, torchaudio, numpy, matplotlib, pandas
    - librosa (optional, for accurate mel->Hz mapping). If missing, a linear freq mapping is used.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio

from models import AudioCNN, compute_conv1_centers


def load_checkpoint_model(ckpt_path: str, device: torch.device, base_channels: int = 32, n_mels: int = 128, n_classes: int = 50):
    model = AudioCNN(n_mels=n_mels, n_classes=n_classes, base_channels=base_channels)
    ck = torch.load(ckpt_path, map_location=device)
    state = ck.get("model_state", ck)
    # DataParallel prefix
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k.replace("module.", "")] = v
            else:
                new_state[k] = v
        try:
            model.load_state_dict(new_state, strict=True)
        except Exception:
            model.load_state_dict(new_state, strict=False)
    else:
        # state is raw tensor? try direct load
        try:
            model.load_state_dict(state)
        except Exception as e:
            raise RuntimeError(f"Failed to load state dict from {ckpt_path}: {e}")
    model.to(device).eval()
    return model

def mel_bin_centers_hz(n_mels: int, sr: int):
    # Return center frequencies for mel bins
    try:
        import librosa
        freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
        return freqs
    except Exception:
        return np.linspace(0.0, sr / 2.0, num=n_mels)

def center_norm_to_hz(center_norm: float, kH: int, mel_centers: np.ndarray) -> float:
    # Map normalized center (0..1) across kH kernel freq bins to frequency in Hz using interpolation
    if len(mel_centers) != kH:
        orig_idx = np.linspace(0, kH - 1, num=len(mel_centers))
        new_idx = np.arange(kH)
        mel_centers = np.interp(new_idx, orig_idx, mel_centers)

    idx = center_norm * (kH - 1)
    idxs = np.arange(kH)
    hz = float(np.interp(idx, idxs, mel_centers))
    return hz

def probe_conv1_with_sines(model: torch.nn.Module, device: torch.device, sr: int = 44100, n_mels: int = 128, n_freqs: int = 48, duration: float = 1.0):
    # Probe conv1 responses using sine tones spaced log-linearly between 100Hz and sr/2
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=n_mels
    )
    
    mel = mel.to(device)
    db = torchaudio.transforms.AmplitudeToDB(stype="power")

    out_ch, in_ch, kH, kW = model.conv1.weight.shape
    freqs = np.geomspace(100.0, sr / 2.0 - 1.0, num=n_freqs)  # avoid exactly Nyquist

    tuning = np.zeros((out_ch, n_freqs), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i, f in enumerate(freqs):
            # synth sine at freq f
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            sine = 0.5 * np.sin(2 * np.pi * f * t).astype(np.float32)
            wav = torch.from_numpy(sine).unsqueeze(0)  # (1, T)
            wav = wav.to(device)
            mel_spec = mel(wav)  # (1, n_mels, T) or (n_mels, T) depend
            if mel_spec.dim() == 3:
                mel_spec = mel_spec.squeeze(0)  # (n_mels, T)
            mel_db = db(mel_spec)
            spec = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            inp = spec.unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,T)
            conv1_out = model.conv1(inp)  # (1, out_ch, H', W')
            # aggregate activation per channel
            act = conv1_out.abs().mean(dim=(2, 3)).squeeze(0).cpu().numpy()  # (out_ch,)
            tuning[:, i] = act
    return tuning, freqs

def save_csv_summary(out_csv: str, centers_norm: np.ndarray, centers_hz: np.ndarray, pref_freqs: np.ndarray):
    df = pd.DataFrame({
        "channel": np.arange(len(centers_norm)),
        "center_norm": centers_norm,
        "center_hz": centers_hz,
        "preferred_hz": pref_freqs,
    })
    df.to_csv(out_csv, index=False)
    print("[INFO] Saved CSV summary:", out_csv)

def plot_centers(centers_hz: np.ndarray, outpath: str, label: str = ""):
    plt.figure(figsize=(8, 4))
    plt.scatter(np.arange(len(centers_hz)), centers_hz, c=np.arange(len(centers_hz)), cmap="viridis")
    plt.xlabel("Conv1 output channel")
    plt.ylabel("Center frequency (Hz)")
    plt.title(f"Conv1 filter centers {label}")
    plt.colorbar(label="channel index")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print("[INFO] Saved centers plot:", outpath)

def plot_tuning_heatmap(tuning: np.ndarray, freqs: np.ndarray, outpath: str, title: str = ""):
    plt.figure(figsize=(8, 6))
    plt.imshow(tuning, aspect="auto", origin="lower", interpolation="nearest")
    plt.xlabel("Frequency index")
    plt.ylabel("Conv1 output channel")
    plt.title(title)
    nfreqs = freqs.shape[0]
    xt = np.linspace(0, nfreqs - 1, min(8, nfreqs)).astype(int)
    plt.xticks(xt, [f"{int(freqs[i])}" for i in xt])
    plt.colorbar(label="activation (a.u.)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print("[INFO] Saved tuning heatmap:", outpath)

def main():
    p = argparse.ArgumentParser(description="Analyze tonotopy from saved models")
    p.add_argument("--neuro-ckpt", type=str, default=None, help="Path to neuro model checkpoint (optional)")
    p.add_argument("--baseline-ckpt", type=str, default=None, help="Path to baseline model checkpoint (optional)")
    p.add_argument("--outdir", type=str, default="./analysis", help="Where to save plots/CSVs")
    p.add_argument("--n-freqs", type=int, default=48, help="Number of probe frequencies")
    p.add_argument("--duration", type=float, default=1.0, help="Duration of synthesized sine (seconds)")
    p.add_argument("--sr", type=int, default=44100, help="Sample rate for probes")
    p.add_argument("--n-mels", type=int, default=128, help="n_mels used at train time (for mapping)")
    p.add_argument("--base-channels", type=int, default=32, help="base channels used in model conv1")
    p.add_argument("--num-classes", type=int, default=50)
    args = p.parse_args()

    ensure_dir = None
    def _ensure(d):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    _ensure(args.outdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    mel_centers = mel_bin_centers_hz(args.n_mels, args.sr)

    for tag, ckpt in (("neuro", args.neuro_ckpt), ("baseline", args.baseline_ckpt)):
        if ckpt is None:
            continue
        print(f"[INFO] Loading {tag} checkpoint: {ckpt}")
        model = load_checkpoint_model(ckpt, device, base_channels=args.base_channels, n_mels=args.n_mels, n_classes=args.num_classes)
        conv1_w = model.conv1.weight.detach().cpu()  # (out, in, kH, kW)
        kH = conv1_w.shape[2]

        centers_norm = compute_conv1_centers(conv1_w.to(device)).cpu().numpy()  # (out,)
        centers_hz = np.array([center_norm_to_hz(c, kH, mel_centers) for c in centers_norm])

        # probe tuning
        tuning, freqs = probe_conv1_with_sines(model, device, sr=args.sr, n_mels=args.n_mels, n_freqs=args.n_freqs, duration=args.duration)
        tuning_norm = tuning / (tuning.max(axis=1, keepdims=True) + 1e-12)

        # preferred freq from tuning
        pref_idx = tuning.argmax(axis=1)
        pref_freqs = freqs[pref_idx]

        # compute smoothness
        log_centers = np.log1p(centers_hz)
        diffs = log_centers[1:] - log_centers[:-1]
        smoothness_mse = float((diffs ** 2).mean())
        print(f"[INFO] {tag}: smoothness MSE (log-centers adjacent) = {smoothness_mse:.6g}")

        csv_path = os.path.join(args.outdir, f"{tag}_conv1_summary_{int(time.time())}.csv")
        save_csv_summary(csv_path, centers_norm, centers_hz, pref_freqs)

        centers_png = os.path.join(args.outdir, f"{tag}_centers_{int(time.time())}.png")
        plot_centers(centers_hz, centers_png, label=tag)

        heat_png = os.path.join(args.outdir, f"{tag}_tuning_heat_{int(time.time())}.png")
        plot_tuning_heatmap(tuning_norm, freqs, heat_png, title=f"{tag} tuning curves (norm per channel)")

        plt.figure(figsize=(8, 4))
        plt.scatter(np.arange(len(pref_freqs)), pref_freqs, c=np.arange(len(pref_freqs)), cmap="plasma")
        plt.xlabel("Conv1 output channel")
        plt.ylabel("Preferred freq (Hz)")
        plt.title(f"Preferred freq by channel ({tag})")
        plt.colorbar(label="channel")
        pref_png = os.path.join(args.outdir, f"{tag}_preferred_freqs_{int(time.time())}.png")
        plt.tight_layout()
        plt.savefig(pref_png, dpi=160)
        plt.close()
        print(f"[INFO] Saved preferred freq scatter: {pref_png}")

    print("[INFO] Analysis completed. Check", args.outdir)

if __name__ == "__main__":
    main()
