import math
import numpy as np
import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    # Input: (B,1,n_mels,T)
    def __init__(self, n_mels=128, n_classes=50, base_channels=32):
        super().__init__()
        c = base_channels
        self.conv1 = nn.Conv2d(1, c, kernel_size=(9,7), padding=(4,3), bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c*2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c*2)
        self.conv3 = nn.Conv2d(c*2, c*4, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c*4)
        self.conv4 = nn.Conv2d(c*4, c*8, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(c*8)
        self.pool = nn.MaxPool2d((2,2))
        self.act = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(c*8, n_classes)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.act(self.bn4(self.conv4(x)))
        x = self.global_pool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x


# Gabor init
def gabor(kH: int, kW: int, theta: float, sigma: float = 1.5, lambd: float = 3.0, gamma: float = 0.5, psi: float = 0.0):
    half_h = kH // 2
    half_w = kW // 2
    y = np.arange(-half_h, half_h + 1)[:, None]
    x = np.arange(-half_w, half_w + 1)[None, :]
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)
    gb = np.exp(-0.5 * (x_theta ** 2 + (gamma * y_theta) ** 2) / (sigma ** 2)) * np.cos(2 * math.pi * x_theta / lambd + psi)
    return gb.astype(np.float32)

def apply_gabor_to_conv(conv: nn.Conv2d, orientations: int = 8):
    out_c, in_c, kH, kW = conv.weight.shape
    oris = np.linspace(0, math.pi, orientations, endpoint=False)
    kernels = np.zeros((out_c, in_c, kH, kW), dtype=np.float32)
    for i in range(out_c):
        theta = oris[i % len(oris)]
        g = gabor(kH, kW, theta)
        for ch in range(in_c):
            kernels[i, ch] = g * (1.0 + 0.02 * np.random.randn(kH, kW))
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(kernels))
    print("[INFO] Applied Gabor init to conv.")


# tonotopy-related helpers
def compute_conv1_centers(conv_weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # conv_weight: (out, in, kH, kW)
    w = conv_weight.detach()
    profile = w.abs().sum(dim=(1, 3))  # out x kH
    denom = profile.sum(dim=1, keepdim=True) + eps
    kH = profile.size(1)
    positions = torch.linspace(0.0, 1.0, steps=kH, device=w.device).unsqueeze(0)
    centers = (profile * positions).sum(dim=1) / denom.squeeze(1)
    return centers  # (out,)

def tonotopy_smoothness_loss(conv_weight: torch.Tensor) -> torch.Tensor:
    centers = compute_conv1_centers(conv_weight)
    diffs = centers[1:] - centers[:-1]
    return diffs.pow(2).mean()
