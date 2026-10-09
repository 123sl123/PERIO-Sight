# -*- coding: utf-8 -*-
"""YOLO11-OBB Mamba modules with parser-compatible C3k2 arguments and integer-only Conv2d groups. These modules do not alter the OBB head, loss or target format and are activated only when explicitly selected in the architecture YAML."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except Exception as e:
    Mamba = None
    _MAMBA_IMPORT_ERROR = e

from .block import C3k2


def _make_divisible(v, divisor=8):
    return int((v + divisor / 2) // divisor * divisor)


def _sanitize_c3k2_args(c1, c2, n, c3k, e, g, shortcut):
    """Normalize parser-supplied C3k2 arguments and ensure Conv2d groups is an integer, not a boolean."""
    if isinstance(c1, bool) or isinstance(c2, bool):
        raise TypeError(f"C3k2Mamba argument parsing error: c1={c1}, c2={c2}. Check that C3k2Mamba is registered in base_modules.")

    c1 = int(c1)
    c2 = int(c2)
    n = int(n)

    # c3k must be a boolean.
    if not isinstance(c3k, bool):
        # Some callers supply c3k as 0 or 1.
        if isinstance(c3k, (int, float)):
            c3k = bool(c3k)
        else:
            c3k = False

    # e must be a float.
    if isinstance(e, bool):
        # Restore the default when e is incorrectly supplied as a bool.
        e = 0.5
    else:
        e = float(e)

    # g must be an integer, not a boolean.
    if isinstance(g, bool):
        g = 1
    else:
        g = int(g)

    if g < 1:
        g = 1

    # shortcut must be a boolean.
    if not isinstance(shortcut, bool):
        shortcut = bool(shortcut)

    return c1, c2, n, c3k, e, g, shortcut


class ChannelSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        channels = int(channels)
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class SpatialMamba2DLite(nn.Module):
    def __init__(
        self,
        channels,
        mamba_ratio=0.5,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        downsample=1,
        layer_scale_init=1e-3,
        use_se=True,
    ):
        super().__init__()

        if Mamba is None:
            raise ImportError(
                f"mamba_ssm Import failed, Ensure the dependencies are installed mamba-ssm and causal-conv1d.Original error: {_MAMBA_IMPORT_ERROR}"
            )

        self.channels = int(channels)
        self.downsample = int(downsample)

        hidden = int(self.channels * float(mamba_ratio))
        hidden = max(64, hidden)
        hidden = min(self.channels, hidden)
        hidden = _make_divisible(hidden, 8)
        self.hidden = int(hidden)

        self.reduce = nn.Conv2d(self.channels, self.hidden, 1, bias=False)
        self.norm = nn.LayerNorm(self.hidden)

        self.mamba = Mamba(
            d_model=self.hidden,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Conv2d(self.hidden, self.channels, 1, bias=False)

        self.gamma = nn.Parameter(layer_scale_init * torch.ones(self.channels))
        self.se = ChannelSE(self.channels) if use_se else nn.Identity()
        self._cpu_warning_emitted = False

    def _apply(self, fn):
        """Keep selective-scan parameters in FP32 when the parent model is cast to FP16."""
        super()._apply(fn)
        return super()._apply(lambda t: t.float() if t.is_floating_point() else t)

    def _run_mamba(self, feat):
        b, c, h, w = feat.shape
        seq = feat.flatten(2).transpose(1, 2).contiguous()

        seq = self.norm(seq).clamp(min=-10.0, max=10.0)
        y1 = self.mamba(seq)

        seq_rev = torch.flip(seq, dims=[1])
        y2 = self.mamba(seq_rev)
        y2 = torch.flip(y2, dims=[1])

        y = 0.5 * (y1 + y2)
        y = self.drop(y)
        y = y.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return y

    def forward(self, x):
        # Ultralytics uses a CPU dummy forward pass to infer strides during construction.
        # mamba_ssm selective_scan requires CUDA inputs.
        # Bypass Mamba on CPU during model construction and return the original features.
        if not x.is_cuda:
            return x

        identity = x
        _, _, h, w = x.shape

        # selective_scan can overflow in FP16 on long spatial sequences.
        # Keep the Mamba branch in FP32 while allowing the rest of YOLO to use AMP.
        with torch.autocast(device_type="cuda", enabled=False):
            z = self.reduce(x.float())

            if self.downsample > 1:
                z_small = F.avg_pool2d(z, kernel_size=self.downsample, stride=self.downsample)
            else:
                z_small = z

            y = self._run_mamba(z_small)

            if self.downsample > 1:
                y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

            y = self.proj(y)
            y = self.se(y)
            scale = 0.1 * torch.tanh(self.gamma / 0.1)
            y = y * scale.view(1, -1, 1, 1)

            # Some Mamba kernels can emit an invalid residual during inference warmup.
            # Keep the trained detector usable by bypassing only that residual branch.
            if not torch.isfinite(y).all():
                return identity

        return identity + y.to(dtype=identity.dtype)


class C3k2Mamba(C3k2):
    """C3k2 with a full Mamba branch for intermediate and deep feature maps."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        c1, c2, n, c3k, e, g, shortcut = _sanitize_c3k2_args(c1, c2, n, c3k, e, g, shortcut)

        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            c3k=c3k,
            e=e,
            g=g,
            shortcut=shortcut,
        )

        self.mamba = SpatialMamba2DLite(
            channels=c2,
            mamba_ratio=0.5,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.0,
            downsample=1,
            layer_scale_init=1e-3,
            use_se=True,
        )

    def forward(self, x):
        x = super().forward(x)
        return self.mamba(x)


class C3k2MambaLite(C3k2):
    """C3k2 with a lightweight Mamba branch for lower-channel or higher-resolution feature maps."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        c1, c2, n, c3k, e, g, shortcut = _sanitize_c3k2_args(c1, c2, n, c3k, e, g, shortcut)

        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            c3k=c3k,
            e=e,
            g=g,
            shortcut=shortcut,
        )

        self.mamba = SpatialMamba2DLite(
            channels=c2,
            mamba_ratio=0.25,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.0,
            downsample=2,
            layer_scale_init=1e-3,
            use_se=True,
        )

    def forward(self, x):
        x = super().forward(x)
        return self.mamba(x)


class AKCConv(nn.Module):
    """
    Paper-inspired adaptive-kernel convolution.

    The CVPR AKCMamba-YOLO paper uses learnable sampling offsets. To keep this
    YOLO11-OBB implementation dependency-free and stable, we approximate that
    idea with content-adaptive multi-kernel depthwise branches. The gate chooses
    how much 3x3 / dilated-3x3 / 5x5 context each image needs.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        channels = int(channels)
        hidden = max(channels // reduction, 16)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            ]
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, len(self.branches), 1, bias=True),
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=1)
        y = 0
        for i, branch in enumerate(self.branches):
            y = y + branch(x) * weights[:, i : i + 1]
        return self.act(self.bn(self.pointwise(y)))


class AKCBlock(nn.Module):
    """AKCBlock from the paper: adaptive local extraction plus optional residual."""

    def __init__(self, channels, shortcut=True):
        super().__init__()
        channels = int(channels)
        self.shortcut = bool(shortcut)
        self.block = nn.Sequential(
            AKCConv(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.shortcut else y


class AKCAttention(nn.Module):
    """Spatial-channel recalibration used after AKSS2D."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        channels = int(channels)
        self.channel = ChannelSE(channels, reduction=reduction)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.channel(x)
        avg = y.mean(dim=1, keepdim=True)
        mx = y.amax(dim=1, keepdim=True)
        return y * self.spatial(torch.cat((avg, mx), dim=1))


class AKSS2D(nn.Module):
    """
    AKSS2D: adaptive-kernel preprocessing + four-direction selective scanning.

    Instead of flattening H*W into one very long sequence, this scans rows and
    columns in forward/backward directions. That is closer to the paper's 2D
    scan idea and is more stable on high-resolution OBB features.
    """

    def __init__(
        self,
        channels,
        mamba_ratio=0.25,
        d_state=16,
        d_conv=4,
        expand=2,
        downsample=1,
        layer_scale_init=1e-3,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                f"mamba_ssm Import failed, Ensure the dependencies are installed mamba-ssm and causal-conv1d.Original error: {_MAMBA_IMPORT_ERROR}"
            )

        self.channels = int(channels)
        self.downsample = int(downsample)
        hidden = _make_divisible(max(64, min(self.channels, int(self.channels * float(mamba_ratio)))), 8)
        self.hidden = int(hidden)

        self.pre = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden, 1, bias=False),
            nn.BatchNorm2d(self.hidden),
            nn.SiLU(inplace=True),
            AKCConv(self.hidden),
        )
        self.norm = nn.LayerNorm(self.hidden)
        self.mamba = Mamba(
            d_model=self.hidden,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        self.proj = nn.Conv2d(self.hidden, self.channels, 1, bias=False)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(self.channels))

    def _apply(self, fn):
        super()._apply(fn)
        return super()._apply(lambda t: t.float() if t.is_floating_point() else t)

    def _scan(self, seq):
        b, l, c = seq.shape
        seq = self.norm(seq).clamp(min=-10.0, max=10.0)
        return self.mamba(seq)

    def _run_akss2d(self, feat):
        b, c, h, w = feat.shape

        rows = feat.permute(0, 2, 3, 1).reshape(b * h, w, c).contiguous()
        rows_f = self._scan(rows)
        rows_b = torch.flip(self._scan(torch.flip(rows, dims=[1])), dims=[1])
        rows_y = 0.5 * (rows_f + rows_b).reshape(b, h, w, c).permute(0, 3, 1, 2)

        cols = feat.permute(0, 3, 2, 1).reshape(b * w, h, c).contiguous()
        cols_f = self._scan(cols)
        cols_b = torch.flip(self._scan(torch.flip(cols, dims=[1])), dims=[1])
        cols_y = 0.5 * (cols_f + cols_b).reshape(b, w, h, c).permute(0, 3, 2, 1)

        return 0.5 * (rows_y + cols_y)

    def forward(self, x):
        if not x.is_cuda:
            return x

        identity = x
        _, _, h, w = x.shape
        with torch.autocast(device_type="cuda", enabled=False):
            z = self.pre(x.float())
            if self.downsample > 1:
                z_small = F.avg_pool2d(z, kernel_size=self.downsample, stride=self.downsample)
            else:
                z_small = z

            y = self._run_akss2d(z_small)

            if self.downsample > 1:
                y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

            y = self.proj(y)
            scale = 0.1 * torch.tanh(self.gamma / 0.1)
            y = y * scale.view(1, -1, 1, 1)
            if not torch.isfinite(y).all():
                raise FloatingPointError("AKSS2D produced non-finite values in the FP32 residual branch.")

        return identity + y.to(dtype=identity.dtype)


class AKCMambaEnhance(nn.Module):
    """Integrated 3CAKCMamba / 4CAKCMamba enhancement block."""

    def __init__(self, channels, local_depth=3, mamba_ratio=0.25, downsample=1, shortcut=True):
        super().__init__()
        channels = int(channels)
        self.local = nn.Sequential(*[AKCBlock(channels, shortcut=shortcut) for _ in range(int(local_depth))])
        self.akss2d = AKSS2D(channels, mamba_ratio=mamba_ratio, downsample=downsample)
        self.attn = AKCAttention(channels)

    def forward(self, x):
        y = self.local(x)
        y = self.akss2d(y)
        return self.attn(y)


class C3k2AKCMamba3(C3k2):
    """Backbone block: C3k2 + paper-style 3CAKC + AKSS2D + AKCAttention."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        c1, c2, n, c3k, e, g, shortcut = _sanitize_c3k2_args(c1, c2, n, c3k, e, g, shortcut)
        super().__init__(c1=c1, c2=c2, n=n, c3k=c3k, e=e, g=g, shortcut=shortcut)
        downsample = 2 if c2 <= 384 else 1
        self.akc_mamba = AKCMambaEnhance(c2, local_depth=3, mamba_ratio=0.25, downsample=downsample, shortcut=True)

    def forward(self, x):
        return self.akc_mamba(super().forward(x))


class C3k2AKCMamba4(C3k2):
    """Neck block: C3k2 + deeper 4CAKC + AKSS2D + AKCAttention."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        c1, c2, n, c3k, e, g, shortcut = _sanitize_c3k2_args(c1, c2, n, c3k, e, g, shortcut)
        super().__init__(c1=c1, c2=c2, n=n, c3k=c3k, e=e, g=g, shortcut=shortcut)
        downsample = 2 if c2 <= 384 else 1
        self.akc_mamba = AKCMambaEnhance(c2, local_depth=4, mamba_ratio=0.25, downsample=downsample, shortcut=True)

    def forward(self, x):
        return self.akc_mamba(super().forward(x))
