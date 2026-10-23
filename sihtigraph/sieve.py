"""Exact additive multiscale image pyramid, then disjoint 4x4 spatial tiles.

We call these Laplacian-like octave residues, not an exact Fourier bandpass.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def residues(image: torch.Tensor) -> torch.Tensor:
    """Input [1,3,32,32], returns [4,3,32,32]; sum is input."""
    if image.shape != (1,3,32,32): raise ValueError('expected one RGB 32x32 image')
    gs = [image]
    for factor in (2,4,8):
        coarse = F.avg_pool2d(image, factor)
        gs.append(F.interpolate(coarse, size=(32,32), mode='bilinear', align_corners=False))
    return torch.cat([gs[i] - gs[i+1] for i in range(3)] + [gs[3]], dim=0)


def pieces(image: torch.Tensor, replacement: str = 'zero') -> tuple[torch.Tensor, torch.Tensor]:
    """Return base [1,3,32,32], pieces [64,3,32,32].

    For any binary gate a, rendered image is base + Σ a_i*piece_i.
    With every gate on it is the original; with every gate off it is the
    replacement. A mean replacement is a fixed RGB CIFAR training mean.
    """
    if replacement == 'zero':
        base = torch.zeros_like(image)
    elif replacement == 'mean':
        base = torch.tensor([.4914, .4822, .4465],device=image.device,dtype=image.dtype).view(1,3,1,1).expand_as(image).clone()
    else: raise ValueError('replacement must be zero or mean')
    contrast = residues(image) - residues(base)
    out = []
    for band in range(4):
        for row in range(4):
            for col in range(4):
                patch = torch.zeros_like(contrast[band])
                patch[:, row*8:(row+1)*8, col*8:(col+1)*8] = contrast[band, :, row*8:(row+1)*8, col*8:(col+1)*8]
                out.append(patch)
    return base, torch.stack(out, 0)


def render_mask(z: torch.Tensor, base: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
    """z [batch,64] signs ±1. Clamp to valid RGB only at classifier boundary."""
    gate = (z.to(parts.dtype) + 1) / 2
    image = base.view(1,-1) + gate @ parts.reshape(64,-1)
    return image.view(-1,3,32,32)
