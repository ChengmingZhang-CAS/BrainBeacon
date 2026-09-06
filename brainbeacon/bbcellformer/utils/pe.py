import torch
from torch import nn
from ..utils import create_norm
import math


def select_pe_encoder(pe):
    if pe in ['sin', 'sinu', 'sinusoidal']:
        return Sinusoidal2dPE
    elif pe in ['learnable', 'bin']:
        return Learnable2dPE
    elif pe in ['naive', 'mlp']:
        return NaivePE
    elif pe in ['lap', 'graphlap', 'lappe']:
        return GraphLapPE
    elif pe in ['fourier', 'cont', 'continuous']:
        return Fourier2dPE
    else:
        raise NotImplementedError(f'Unsupported positional encoding type: {pe}')

class Sinusoidal2dPE(nn.Module):
    def __init__(self, d_model, height=100, width=100):
        """
        :param d_model: dimension of the model_raw
        :param height: height of the positions
        :param width: width of the positions
        """
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        self.d_model = d_model
        self.height = height
        self.width = width
        self.pe_key = 'coord'
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)

        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.pe_enc = nn.Embedding.from_pretrained(pe.flatten(1).T)

    def forward(self, coordinates):
        if coordinates[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(coordinates.shape[0], -1)
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        x = ((x*1.02-0.01) * self.width).long()
        y = ((y*1.02-0.01) * self.height).long()
        x[x >= self.width] = self.width - 1
        y[y >= self.height] = self.height - 1
        x[x < 0] = 0
        y[y < 0] = 0
        pe_input = x * self.width + y
        return self.pe_enc(pe_input)

class Learnable2dPE(nn.Module):
    def __init__(self, d_model, height=100, width=100):
        """
        :param d_model: dimension of the model_raw
        :param height: height of the positions
        :param width: width of the positions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.pe_enc = nn.Embedding(height * width, d_model)
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)
        self.pe_key = 'coord'

    def forward(self, coordinates):
        if coordinates[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(coordinates.shape[0], -1)
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        x = ((x*1.02-0.01) * self.width).long()
        y = ((y*1.02-0.01) * self.height).long()
        x[x >= self.width] = self.width - 1
        y[y >= self.height] = self.height - 1
        x[x < 0] = 0
        y[y < 0] = 0
        pe_input = x * self.width + y
        return self.pe_enc(pe_input)

class NaivePE(nn.Module):
    def __init__(self, d_model, coord_dim = 2, height=None, width=None):
        """
        :param d_model: dimension of the model_raw
        :param coord_dim: dimension of coordinates
        :param height: placeholder
        :param width: placeholder
        """
        super().__init__()
        self.pe_enc = nn.Sequential(
                            nn.Linear(coord_dim, d_model),
                            nn.PReLU(),
        )
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)
        self.pe_key = 'coord'

    def forward(self, coordinates):
        if coordinates[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(coordinates.shape[0], -1)
        return self.pe_enc(coordinates)

class GraphLapPE(nn.Module):
    def __init__(self, d_model, k = 10, height=None, width=None):
        """
        :param d_model: dimension of the model_raw
        :param k: top k
        :param height: placeholder
        :param width: placeholder
        """
        super().__init__()
        self.k = k
        self.pe_enc = nn.Sequential(
                            nn.Linear(k, d_model),
                            nn.PReLU(),
        )
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)
        self.pe_key = 'eigvec'

    def forward(self, eigvec):
        if eigvec[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(eigvec.shape[0], -1)
        eigvec = eigvec * (torch.randint(0, 2, (self.k, ), dtype=torch.float, device=eigvec.device)[None, :]*2-1)
        return self.pe_enc(eigvec)


class Fourier2dPE(nn.Module):
    """
    Continuous Fourier features for 2D coordinates in [0,1].

    This version is tuned for locality (recommended default):
      - lower max_freq to reduce oscillation/aliasing
      - apply frequency attenuation so low-freq dominates (more monotone-like locally)
      - keep interface unchanged: forward(coords)->(N, d_model)
      - NOT a 100x100 grid lookup (fully continuous)
    """
    def __init__(
        self,
        d_model: int,
        max_freq: float = 8.0,
        freq_scale: str = "log",          # "log" or "linear"
        weight_type: str = "inv",         # "none" | "inv" | "exp"
        weight_alpha: float = 1.0,        # inv: 1/(f^alpha), exp: exp(-alpha*f)
        normalize_weight: bool = True,    # keep overall magnitude stable
        eps: float = 1e-12,
        height=None,
        width=None,
    ):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"Fourier2dPE requires d_model % 4 == 0, got {d_model}")

        self.pe_key = "coord"
        self.missing_pe = nn.Parameter(torch.randn(d_model) * 1e-2)

        half = d_model // 2               # x part + y part
        n_freq = half // 2                # sin+cos pairs per axis

        # Frequencies
        if freq_scale == "log":
            freqs = torch.logspace(0, math.log10(max_freq), steps=n_freq)
        elif freq_scale == "linear":
            freqs = torch.linspace(1.0, max_freq, steps=n_freq)
        else:
            raise ValueError(f"Unsupported freq_scale={freq_scale}")

        # Frequency attenuation (locality-friendly)
        if weight_type == "none":
            w = torch.ones_like(freqs)
        elif weight_type == "inv":
            w = 1.0 / (freqs.clamp(min=eps) ** float(weight_alpha))
        elif weight_type == "exp":
            w = torch.exp(-float(weight_alpha) * freqs)
        else:
            raise ValueError(f"Unsupported weight_type={weight_type}")

        if normalize_weight:
            w = w / (w.mean().clamp(min=eps))

        self.register_buffer("freqs", freqs)   # (n_freq,)
        self.register_buffer("freq_w", w)      # (n_freq,)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates[0][0] == -1:
            return self.missing_pe.unsqueeze(0).expand(coordinates.shape[0], -1)

        x = coordinates[:, 0:1]  # (N,1)
        y = coordinates[:, 1:2]

        wx = x * (2 * math.pi) * self.freqs[None, :]  # (N, n_freq)
        wy = y * (2 * math.pi) * self.freqs[None, :]

        w = self.freq_w[None, :]  # (1, n_freq)

        pe_x = torch.cat([torch.sin(wx) * w, torch.cos(wx) * w], dim=1)  # (N, 2*n_freq)
        pe_y = torch.cat([torch.sin(wy) * w, torch.cos(wy) * w], dim=1)  # (N, 2*n_freq)
        return torch.cat([pe_x, pe_y], dim=1)                            # (N, d_model)