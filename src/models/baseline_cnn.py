import torch
from torch import nn


class BaselineCNN1d(nn.Module):
    """A simple local 1D CNN baseline for u(x, 0) -> u(x, T).

    Input shape: [batch, n_grid].
    Output shape: [batch, n_grid].

    Unlike FNO, this baseline only mixes nearby grid points through convolution
    kernels. Circular padding keeps the periodic boundary condition visible to
    the model, because the first and last grid points should be neighbors.
    """

    def __init__(self, channels: int = 32, layers: int = 4, kernel_size: int = 5) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if layers <= 0:
            raise ValueError("layers must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        padding = kernel_size // 2
        modules: list[nn.Module] = [
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding=padding, padding_mode="circular"),
            nn.GELU(),
        ]

        for _ in range(layers - 1):
            modules.extend(
                [
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        padding_mode="circular",
                    ),
                    nn.GELU(),
                ]
            )

        modules.append(nn.Conv1d(channels, 1, kernel_size=1))
        self.net = nn.Sequential(*modules)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.dim() != 2:
            raise ValueError(f"BaselineCNN1d expects input shape [batch, n_grid], got {u.shape}")

        # Conv1d expects [batch, channels, n_grid], so add a single field channel.
        x = u.unsqueeze(1)
        x = self.net(x)
        return x.squeeze(1)
