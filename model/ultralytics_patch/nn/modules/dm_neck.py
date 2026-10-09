"""Neck-only DySample upsampling used by the YOLO-DM ablation models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Content-adaptive 2x upsampling without changing the feature channels."""

    def __init__(self, scale=2, groups=4, offset_scale=0.25):
        super().__init__()
        self.scale = int(scale)
        self.groups = int(groups)
        self.offset_scale = float(offset_scale)
        self.offset = None

    def _build(self, channels, device):
        self.offset = nn.Conv2d(channels, 2 * self.groups * self.scale**2, 1, bias=True).to(device)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x):
        if self.offset is None:
            self._build(x.shape[1], x.device)

        y = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        b, _, h, w = y.shape
        offsets = F.pixel_shuffle(self.offset(x), self.scale).view(b, self.groups, 2, h, w).mean(dim=1)
        offsets = torch.tanh(offsets) * self.offset_scale

        gy = torch.linspace(-1.0, 1.0, h, device=y.device, dtype=y.dtype)
        gx = torch.linspace(-1.0, 1.0, w, device=y.device, dtype=y.dtype)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        delta = torch.stack((offsets[:, 0] * 2.0 / max(w - 1, 1), offsets[:, 1] * 2.0 / max(h - 1, 1)), dim=-1)
        return F.grid_sample(y, (base_grid + delta).clamp(-1.0, 1.0), mode="bilinear", padding_mode="border", align_corners=True)
