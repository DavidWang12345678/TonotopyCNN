"""
Visualize conv1 centers and compute tuning curves by probing conv1 with pure tones
Usage:
    python visualize.py --ckpt path/to/best_neuro.pt --outdir ./viz --n_freqs 40
    python visualize.py --ckpt path/to/best_baseline.pt --outdir ./viz --n_freqs 40
"""
import os
import argparse
import torch
import numpy as np
import time
from utils import ensure_dir, make_sine, plot_tuning_curves
from models import AudioCNN, compute_conv1_centers

import torchaudio

def load_model_checkpoint(ckpt_path, device='cpu', n_mels=128, n_classes=50, base_channels=32):
    model = AudioCNN(n_mels=n_mels, n_classes=n_classes, base_channels=base_channels)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("model_state", state))
    model.to(device).eval()
    return model

def probe_conv1_tuning(model, device='cpu', sr=44100, n_mels=128, n_freqs=40, duration=1.0):
    # Generate sine waves across log frequencies, convert to mel spectrogram, pass through conv1 and pool to obtain activation per channel
    freqs = np.geomspace(100, 8000, num=n_freqs)
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    db = torchaudio.transforms.AmplitudeToDB(stype="power")
    out_ch = model.conv1.weight.shape[0]
    tuning = np.zeros((out_ch, n_freqs), dtype=np.float32)
    for i, f in enumerate(freqs):
        sine = make_sine(f, duration=duration, sr=sr)
        with torch.no_grad():
            wav = sine.to(device)
            mel = mel_transform(wav)
            if mel.dim() == 3:
                mel = mel.squeeze(0)
            mel_db = db(mel)
            spec = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            inp = spec.unsqueeze(0).unsqueeze(0).to(device)
            conv1 = model.conv1(inp)
            act = conv1.abs().mean(dim=(2,3)).cpu().numpy().squeeze()
            tuning[:, i] = act
    return tuning, freqs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./viz")
    parser.add_argument("--n-freqs", type=int, default=40, dest="n_freqs")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sr", type=int, default=44100)
    args = parser.parse_args()
    ensure_dir(args.outdir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_checkpoint(args.ckpt, device=device)
    tuning, freqs = probe_conv1_tuning(model, device=device, sr=args.sr, n_mels=128, n_freqs=args.n_freqs, duration=args.duration)
    outpath = os.path.join(args.outdir, f"tuning_{int(time.time())}.png")
    plot_tuning_curves(tuning, freqs, outpath, title="Conv1 tuning curves")
    print("Saved tuning curves to", outpath)
    centers = compute_conv1_centers(model.conv1.weight).cpu().numpy()
    np.save(os.path.join(args.outdir, "conv1_centers.npy"), centers)
    print("Saved conv1 centers.")
