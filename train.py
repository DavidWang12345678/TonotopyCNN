import os
import json
import argparse
from datetime import datetime
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datasets import load_dataset

from utils import set_seed, ensure_dir, make_mel_transform, save_metrics, plot_conv1_centers_and_profiles, make_sine
from models import AudioCNN, apply_gabor_to_conv, tonotopy_smoothness_loss

import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_grad_scaler(use_amp: bool, device_type: str):
    from torch import amp
    try:
        return amp.GradScaler(device_type=device_type, enabled=use_amp)
    except TypeError:
        try:
            return amp.GradScaler(enabled=use_amp)
        except Exception:
            return None
from torch import amp
def autocast_ctx(use_amp):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return amp.autocast(device_type=device_type, enabled=use_amp)

# Dataset wrapper using HuggingFace
class ESC50HFDatasetTorch:
    def __init__(self, hf_dataset, folds=None, sample_rate=44100, n_mels=128, n_fft=2048, hop_length=512, augment=False):
        self.hf = hf_dataset
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment
        self.indices = [i for i, ex in enumerate(self.hf) if (folds is None or ex.get("fold") in folds)]
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")
        if augment:
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        else:
            self.time_mask = None
            self.freq_mask = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real = self.indices[idx]
        ex = self.hf[real]
        audio = ex["audio"]
        arr = audio["array"]
        sr = audio["sampling_rate"]
        wav = torch.tensor(arr).unsqueeze(0).float()
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        mel = self.mel(wav)
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        mel_db = self.db(mel)
        spec = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        if self.augment:
            spec = spec.unsqueeze(0)
            if self.freq_mask:
                spec = self.freq_mask(spec)
            if self.time_mask:
                spec = self.time_mask(spec)
            spec = spec.squeeze(0)
        return spec.unsqueeze(0).float(), int(ex["target"])


def train_and_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    set_seed(args.seed)
    ensure_dir(args.save_dir)
    print("[INFO] Device:", device)

    print("[INFO] Loading ESC-50 from Hugging Face (this will download to local cache)...")
    hf = load_dataset("ashraq/esc50", split="train")
    # train folds list
    train_folds = [int(x) for x in args.train_folds.split(",")]
    val_fold = args.val_fold

    train_ds = ESC50HFDatasetTorch(hf, folds=train_folds, sample_rate=args.sample_rate, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length, augment=args.augment)
    val_ds = ESC50HFDatasetTorch(hf, folds=[val_fold], sample_rate=args.sample_rate, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length, augment=False)

    if args.smoke:
        train_ds = Subset(train_ds, list(range(min(64, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(64, len(val_ds)))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"[INFO] train samples {len(train_loader.dataset)}, val samples {len(val_loader.dataset)}")

    model = AudioCNN(n_mels=args.n_mels, n_classes=args.num_classes, base_channels=args.base_channels)
    if args.arch == "neuro" and args.gabor_init:
        apply_gabor_to_conv(model.conv1, orientations=args.gabor_orientations)
    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    device_type = "cuda" if device.type == "cuda" else "mps"
    use_amp = args.use_amp and device.type == "cuda"
    from torch import amp
    scaler = make_grad_scaler(use_amp, device_type)

    best_val = -1.0
    metrics = []

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} (train)", leave=False)
        for spec, label in pbar:
            spec = spec.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp):
                logits = model(spec)
                loss = criterion(logits, label)
                if args.arch == "neuro" and args.lambda_tono > 0.0:
                    loss = loss + args.lambda_tono * tonotopy_smoothness_loss(model.conv1.weight)

            if scaler is not None:
                try:
                    scaler.scale(loss).backward()
                    if hasattr(scaler, "unscale_"):
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                except Exception:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            total += spec.size(0)
            running_loss += float(loss.item()) * spec.size(0)
            preds = logits.argmax(dim=1)
            running_acc += 100.0 * (preds == label).float().mean().item() * spec.size(0)
            pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{running_acc/total:.2f}"})

        train_loss = running_loss / total
        train_acc = running_acc / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        vloss = 0.0
        vacc = 0.0
        vtotal = 0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} (val)", leave=False)
            for spec, label in vbar:
                spec = spec.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                with autocast_ctx(use_amp):
                    logits = model(spec)
                    loss = criterion(logits, label)
                vtotal += spec.size(0)
                vloss += float(loss.item()) * spec.size(0)
                preds = logits.argmax(dim=1)
                vacc += 100.0 * (preds == label).float().mean().item() * spec.size(0)
                vbar.set_postfix({"vloss": f"{vloss/vtotal:.4f}", "vacc": f"{vacc/vtotal:.2f}"})

        val_loss = vloss / max(1, vtotal)
        val_acc = vacc / max(1, vtotal)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        metrics.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val:
            best_val = val_acc
            ckpt = os.path.join(args.save_dir, f"best_{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
            torch.save({"model_state": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch, "val_acc": val_acc}, ckpt)
            print("[INFO] New best model saved:", ckpt)

        print(f"Epoch {epoch}/{args.epochs}  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%   Val: loss={val_loss:.4f}, acc={val_acc:.2f}%")

        if epoch % args.plot_every == 0 or epoch == args.epochs:
            conv1 = model.conv1.weight.detach().cpu()
            p = os.path.join(args.save_dir, f"{args.arch}_conv1_ep{epoch}.png")
            plot_conv1_centers_and_profiles(conv1, p, title=f"{args.arch} ep{epoch}")
            print("[INFO] conv1 visualization saved:", p)

    save_metrics(metrics, args.save_dir, f"metrics_{args.arch}")
    
    metrics_dict = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }
    metrics_path = os.path.join(args.save_dir, f"metrics_{args.arch}_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f)
    print(f"[INFO] Saved metrics for {args.arch} -> {metrics_path}")

    # If both baseline and neuro metric files exist, produce comparison plot
    baseline_path = os.path.join(args.save_dir, "metrics_baseline_comparison.json")
    neuro_path = os.path.join(args.save_dir, "metrics_neuro_comparison.json")

    if os.path.exists(baseline_path) and os.path.exists(neuro_path):
        print("[INFO] Both models completed — generating comparison plot...")
        with open(baseline_path, "r") as f:
            base = json.load(f)
        with open(neuro_path, "r") as f:
            neuro = json.load(f)

        epochs_base = list(range(1, len(base["val_losses"]) + 1))
        epochs_neuro = list(range(1, len(neuro["val_losses"]) + 1))

        color_base = "#E67E22"  # orange
        color_neuro = "#1F77B4"  # blue

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Training loss comparison
        axes[0, 0].plot(epochs_base, base["train_losses"], label="Baseline", color=color_base, linewidth=2)
        axes[0, 0].plot(epochs_neuro, neuro["train_losses"], label="Neuro-inspired", color=color_neuro, linewidth=2)
        axes[0, 0].set_xlabel("Epoch", fontsize=11)
        axes[0, 0].set_ylabel("Training Loss", fontsize=11)
        axes[0, 0].set_title("Training Loss Comparison", fontsize=13)
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle="--", alpha=0.6)

        # Validation loss comparison
        axes[0, 1].plot(epochs_base, base["val_losses"], label="Baseline", color=color_base, linewidth=2)
        axes[0, 1].plot(epochs_neuro, neuro["val_losses"], label="Neuro-inspired", color=color_neuro, linewidth=2)
        axes[0, 1].set_xlabel("Epoch", fontsize=11)
        axes[0, 1].set_ylabel("Validation Loss", fontsize=11)
        axes[0, 1].set_title("Validation Loss Comparison", fontsize=13)
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle="--", alpha=0.6)

        # Training accuracy comparison
        axes[1, 0].plot(epochs_base, base["train_accs"], label="Baseline", color=color_base, linewidth=2)
        axes[1, 0].plot(epochs_neuro, neuro["train_accs"], label="Neuro-inspired", color=color_neuro, linewidth=2)
        axes[1, 0].set_xlabel("Epoch", fontsize=11)
        axes[1, 0].set_ylabel("Training Accuracy (%)", fontsize=11)
        axes[1, 0].set_title("Training Accuracy Comparison", fontsize=13)
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle="--", alpha=0.6)

        # Validation accuracy comparison
        axes[1, 1].plot(epochs_base, base["val_accs"], label="Baseline", color=color_base, linewidth=2)
        axes[1, 1].plot(epochs_neuro, neuro["val_accs"], label="Neuro-inspired", color=color_neuro, linewidth=2)
        axes[1, 1].set_xlabel("Epoch", fontsize=11)
        axes[1, 1].set_ylabel("Validation Accuracy (%)", fontsize=11)
        axes[1, 1].set_title("Validation Accuracy Comparison", fontsize=13)
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle="--", alpha=0.6)

        for ax in axes.flat:
            ax.set_xlim(1, max(len(epochs_base), len(epochs_neuro)))
        
        # Set y-limits for loss plots
        all_train_losses = base["train_losses"] + neuro["train_losses"]
        all_val_losses = base["val_losses"] + neuro["val_losses"]
        axes[0, 0].set_ylim(0, max(all_train_losses) * 1.1)
        axes[0, 1].set_ylim(0, max(all_val_losses) * 1.1)
        
        # Set y-limits for accuracy plots
        all_train_accs = base["train_accs"] + neuro["train_accs"]
        all_val_accs = base["val_accs"] + neuro["val_accs"]
        axes[1, 0].set_ylim(0, min(100, max(all_train_accs) * 1.1))
        axes[1, 1].set_ylim(0, min(100, max(all_val_accs) * 1.1))

        plt.tight_layout()
        out_path = os.path.join(args.save_dir, "comparison_curves.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved comparison plot to {out_path}")
    else:
        print(f"[INFO] Waiting for both models to complete for comparison plot...")

    print("[INFO] Best val acc:", best_val)


def main():
    p = argparse.ArgumentParser(description="Train tonotopy models on ESC-50 (HuggingFace)")
    p.add_argument("--arch", choices=["baseline","neuro"], default="neuro")
    p.add_argument("--gabor-init", action="store_true")
    p.add_argument("--gabor-orientations", type=int, default=8)
    p.add_argument("--lambda-tono", type=float, default=1e-4, dest="lambda_tono")
    p.add_argument("--n-mels", type=int, default=128, dest="n_mels")
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512, dest="hop_length")
    p.add_argument("--sample-rate", type=int, default=44100, dest="sample_rate")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--plot-every", type=int, default=10, dest="plot_every")
    p.add_argument("--save-dir", type=str, default="./runs_tonotopy", dest="save_dir")
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--clip-grad-norm", type=float, default=5.0, dest="clip_grad_norm")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--train-folds", type=str, default="1,2,3,4")
    p.add_argument("--val-fold", type=int, default=5)
    p.add_argument("--num-classes", type=int, default=50)
    args = parser.parse_args()
    args.train_folds = args.train_folds
    ensure_dir(args.save_dir)
    print("[CONFIG]", json.dumps(vars(args), indent=2))
    train_and_eval(args)

if __name__ == "__main__":
    main()
