import os
import json
import math
import random
import time
from datetime import datetime
from typing import Tuple, List

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn


def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_metrics(metrics: list, out_dir: str, prefix: str):
    ensure_dir(out_dir)
    ts = timestamp()
    csvp = os.path.join(out_dir, f"{prefix}_metrics_{ts}.csv")
    jsonp = csvp.replace(".csv", ".json")
    try:
        pd = __import__("pandas")
        df = pd.DataFrame(metrics)
        df.to_csv(csvp, index=False)
        with open(jsonp, "w") as f:
            json.dump(metrics, f, indent=2)
        print("[INFO] Saved metrics:", csvp, jsonp)
    except Exception as e:
        print("[WARN] Could not save metrics:", e)


# spectrogram transforms
def make_mel_transform(sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    db = torchaudio.transforms.AmplitudeToDB(stype="power")
    return mel, db


# plotting helpers
def plot_conv1_centers_and_profiles(conv_weight: torch.Tensor, outpath: str, title: str = ""):
    ensure_dir(os.path.dirname(outpath))
    w = conv_weight.detach().cpu()
    # compute centers as weighted average across kH
    profile = w.abs().sum(dim=(1, 3))  # out x kH
    denom = profile.sum(dim=1, keepdim=True) + 1e-8
    kH = profile.size(1)
    positions = torch.linspace(0.0, 1.0, steps=kH).unsqueeze(0)
    centers = (profile * positions).sum(dim=1) / denom.squeeze(1)
    centers = centers.numpy()

    num_show = min(16, w.size(0))
    profiles = profile[:num_show].numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(centers, marker="o")
    ax[0].set_title("Conv1 center frequency by channel")
    ax[0].set_xlabel("out channel")
    ax[0].set_ylabel("normalized center (0..1)")
    im = ax[1].imshow(profiles, aspect="auto", interpolation="nearest")
    ax[1].set_title("Conv1 profiles (out x kH)")
    ax[1].set_xlabel("kernel frequency bin")
    fig.colorbar(im, ax=ax[1])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def plot_tuning_curves(tuning_matrix: np.ndarray, freqs: np.ndarray, outpath: str, title: str = ""):
    ensure_dir(os.path.dirname(outpath))
    num = tuning_matrix.shape[0]
    ncols = 4
    nrows = math.ceil(num / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2 * nrows))
    axs = axs.flatten()
    for i in range(num):
        axs[i].plot(freqs, tuning_matrix[i])
        axs[i].set_title(f"ch {i}")
        axs[i].set_xlabel("freq")
    for j in range(num, len(axs)):
        fig.delaxes(axs[j])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def make_sine(freq: float, duration: float = 1.0, sr: int = 44100, amp: float = 0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = amp * np.sin(2 * np.pi * freq * t)
    return torch.from_numpy(x).float().unsqueeze(0)  # (1, T)
