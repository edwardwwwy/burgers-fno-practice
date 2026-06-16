import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv1d(nn.Module):
    """1D spectral convolution used by FNO.

    Input shape: [batch, in_channels, n_grid].
    Output shape: [batch, out_channels, n_grid].
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        if modes <= 0:
            raise ValueError("modes must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, n_grid = x.shape

        # rfft keeps only non-negative frequencies, so the frequency dimension
        # has size n_grid // 2 + 1 instead of n_grid.
        x_ft = torch.fft.rfft(x, dim=-1)
        available_modes = x_ft.shape[-1]
        used_modes = min(self.modes, available_modes)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            available_modes,
            device=x.device,
            dtype=torch.cfloat,
        )

        # Keep only the lowest Fourier modes. This is the main FNO truncation:
        # high-frequency coefficients remain zero in out_ft.
        out_ft[:, :, :used_modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :used_modes],
            self.weights[:, :, :used_modes],
        )

        # Pass n_grid explicitly so irfft reconstructs the original spatial size.
        return torch.fft.irfft(out_ft, n=n_grid, dim=-1)


class FNO1d(nn.Module):
    """Minimal Fourier Neural Operator for u(x, 0) -> u(x, T)."""

    def __init__(self, modes: int, width: int, layers: int = 3) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if layers <= 0:
            raise ValueError("layers must be positive")

        self.modes = modes
        self.width = width
        self.layers = layers

        # Each grid point receives two features: u(x) and its coordinate x.
        self.lift = nn.Linear(2, width)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(layers)]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(layers)]
        )
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Run the FNO forward pass.

        Expected input shape is [batch, n_grid]. The output has the same shape.
        Internally, Conv1d and FFT use [batch, channels, n_grid].
        """
        if u.dim() != 2:
            raise ValueError(f"FNO1d expects input shape [batch, n_grid], got {u.shape}")

        batch_size, n_grid = u.shape
        grid = torch.linspace(0.0, 1.0, n_grid, device=u.device, dtype=u.dtype)
        grid = grid.view(1, n_grid, 1).repeat(batch_size, 1, 1)

        x = torch.cat([u.unsqueeze(-1), grid], dim=-1)
        x = self.lift(x)

        # Switch from pointwise layout [batch, n_grid, channels] to the Conv1d
        # layout [batch, channels, n_grid].
        x = x.permute(0, 2, 1)

        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            x = F.gelu(spectral(x) + pointwise(x))

        x = x.permute(0, 2, 1)
        return self.project(x).squeeze(-1)
