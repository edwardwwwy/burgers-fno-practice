from __future__ import annotations

import numpy as np
import torch


def sample_initial_conditions(
    n_samples: int,
    n_grid: int,
    max_modes: int = 4,
    seed: int | None = None,
) -> np.ndarray:
    """Sample smooth periodic initial conditions on x in [0, 1).

    The output shape is [n_samples, n_grid]. Each sample is a small random
    Fourier series, normalized so the explicit solver stays stable in smoke tests.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False, dtype=np.float64)
    u0 = np.zeros((n_samples, n_grid), dtype=np.float64)

    for mode in range(1, max_modes + 1):
        sin_coeff = rng.normal(0.0, 1.0 / mode, size=(n_samples, 1))
        cos_coeff = rng.normal(0.0, 1.0 / mode, size=(n_samples, 1))
        angle = 2.0 * np.pi * mode * x[None, :]
        u0 += sin_coeff * np.sin(angle) + cos_coeff * np.cos(angle)

    max_abs = np.max(np.abs(u0), axis=1, keepdims=True)
    u0 = 0.5 * u0 / np.maximum(max_abs, 1e-12)
    return u0


def burgers_rhs(u: np.ndarray, dx: float, viscosity: float) -> np.ndarray:
    """Compute the right-hand side of u_t = -u u_x + nu u_xx.

    Shape convention: u has shape [batch, n_grid]. Periodic boundary conditions
    are implemented with np.roll, so the first and last grid points are neighbors.
    """
    u_left = np.roll(u, shift=1, axis=1)
    u_right = np.roll(u, shift=-1, axis=1)

    ux = (u_right - u_left) / (2.0 * dx)
    uxx = (u_right - 2.0 * u + u_left) / (dx * dx)
    return -u * ux + viscosity * uxx


def solve_burgers_periodic(
    u0: np.ndarray,
    viscosity: float,
    time_horizon: float,
    time_steps: int,
) -> np.ndarray:
    """Evolve a batch of periodic Burgers states with a simple RK4 solver.

    This solver is intentionally small and readable. The first project version
    uses tiny grids and short horizons so this CPU implementation is enough for
    smoke tests.
    """
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")

    u = np.array(u0, dtype=np.float64, copy=True)
    n_grid = u.shape[1]
    dx = 1.0 / n_grid
    dt = time_horizon / time_steps

    for _ in range(time_steps):
        k1 = burgers_rhs(u, dx, viscosity)
        k2 = burgers_rhs(u + 0.5 * dt * k1, dx, viscosity)
        k3 = burgers_rhs(u + 0.5 * dt * k2, dx, viscosity)
        k4 = burgers_rhs(u + dt * k3, dx, viscosity)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.isfinite(u).all():
            raise FloatingPointError("Burgers solver produced non-finite values")

    return u


def generate_burgers_data(
    n_samples: int,
    n_grid: int,
    viscosity: float,
    time_horizon: float,
    time_steps: int,
    initial_condition_modes: int = 4,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a tiny supervised dataset for u(x, 0) -> u(x, T).

    Returned tensors both have shape [n_samples, n_grid] and dtype float32.
    """
    u0 = sample_initial_conditions(
        n_samples=n_samples,
        n_grid=n_grid,
        max_modes=initial_condition_modes,
        seed=seed,
    )
    uT = solve_burgers_periodic(
        u0=u0,
        viscosity=viscosity,
        time_horizon=time_horizon,
        time_steps=time_steps,
    )
    return torch.from_numpy(u0).float(), torch.from_numpy(uT).float()
