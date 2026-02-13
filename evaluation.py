"""
Evaluation and reporting utility functions.
"""

import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from physics import MirrorPropagator


def evaluate_fidelity_grid(
    propagator: MirrorPropagator,
    Rabi_i: torch.Tensor,
    Rabi_q: torch.Tensor,
    delta: torch.Tensor,
    N_grid: int = 33,
    dp_range: tuple = (-0.5, 0.5),
    beta_range: tuple = (-0.2, 0.2),
    batch_size: int = 256,
):
    """Evaluate mirror fidelity on an NxN grid of (delta_p, beta).

    Processes the grid in chunks of ``batch_size`` to avoid OOM.

    Returns:
        fidelity: (N_grid, N_grid) numpy array
        dp_grid: (N_grid,) numpy array
        beta_grid: (N_grid,) numpy array
    """
    device = Rabi_i.device
    with torch.no_grad():
        dp_grid = torch.linspace(*dp_range, N_grid, dtype=torch.float64, device=device)
        beta_grid = torch.linspace(
            *beta_range, N_grid, dtype=torch.float64, device=device
        )
        dp_mesh, beta_mesh = torch.meshgrid(dp_grid, beta_grid, indexing="ij")
        dp_flat = dp_mesh.flatten()
        beta_flat = beta_mesh.flatten()

        losses = []
        for i in range(0, dp_flat.shape[0], batch_size):
            dp_batch = dp_flat[i : i + batch_size]
            beta_batch = beta_flat[i : i + batch_size]
            H = propagator.build_hamiltonians(
                Rabi_i, Rabi_q, delta, dp_batch, beta_batch
            )
            U = propagator.compute_propagator(H)
            losses.append(propagator.mirror_loss(U, reduction="none"))

        loss_eval = torch.cat(losses)
        fidelity = 1 - loss_eval.reshape(N_grid, N_grid).cpu().numpy()

    return fidelity, dp_grid.cpu().numpy(), beta_grid.cpu().numpy()


def log_fidelity_summary(fidelity: np.ndarray):
    """Log summary statistics for a fidelity grid."""
    logger.info("Mean fidelity over grid: {:.4f}", fidelity.mean())
    logger.info("Min fidelity over grid:  {:.4f}", fidelity.min())
    mid = fidelity.shape[0] // 2
    logger.info("Fidelity at (delta_p=0, beta=0): {:.6f}", fidelity[mid, mid])
    logger.info("Fidelity > 0.9 fraction:  {:.2f}", (fidelity > 0.9).mean())


def plot_parameters(
    Rabi_i: np.ndarray,
    Rabi_q: np.ndarray,
    delta: np.ndarray,
    time_us: np.ndarray,
    gauss_ref: np.ndarray | None = None,
    save_path: str | None = None,
):
    """Plot all optimization parameters over the time grid.

    Panels: Rabi_i, Rabi_q, Rabi amplitude, laser phase, detuning.
    If ``gauss_ref`` is provided, a dashed Gaussian envelope is overlaid on
    the Rabi_i and Rabi amplitude panels.
    """
    omega_R = np.sqrt(Rabi_i**2 + Rabi_q**2)
    phi_L = np.arctan2(Rabi_q, Rabi_i)

    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

    axes[0].step(time_us, Rabi_i, where="mid", color="C0")
    if gauss_ref is not None:
        axes[0].plot(
            time_us, gauss_ref, "--", color="grey", alpha=0.7, label="Gaussian"
        )
        axes[0].legend(fontsize=8)
    axes[0].set_ylabel(r"$\Omega_I$ (rad/$\mu$s)")
    axes[0].set_title("In-phase Rabi component")

    axes[1].step(time_us, Rabi_q, where="mid", color="C1")
    axes[1].set_ylabel(r"$\Omega_Q$ (rad/$\mu$s)")
    axes[1].set_title("Quadrature Rabi component")

    axes[2].step(time_us, omega_R, where="mid", color="C2")
    if gauss_ref is not None:
        axes[2].plot(
            time_us, gauss_ref, "--", color="grey", alpha=0.7, label="Gaussian"
        )
        axes[2].legend(fontsize=8)
    axes[2].set_ylabel(r"$\Omega_R$ (rad/$\mu$s)")
    axes[2].set_title("Rabi amplitude")

    axes[3].step(time_us, phi_L, where="mid", color="C3")
    axes[3].set_ylabel(r"$\phi_L$ (rad)")
    axes[3].set_title("Rabi phase")

    axes[4].step(time_us, delta, where="mid", color="C4")
    axes[4].set_ylabel(r"$\delta$ (rad/$\mu$s)")
    axes[4].set_title("Detuning")

    axes[4].set_xlabel(r"Time ($\mu$s)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Parameter plot saved to {}", save_path)
    return fig


def plot_fidelity_contour(
    fidelity: np.ndarray,
    dp_grid: np.ndarray,
    beta_grid: np.ndarray,
    title: str = "Mirror fidelity",
    save_path: str | None = None,
):
    """Plot fidelity as a filled contour over the (delta_p, beta) grid."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_fidelity_contour(ax, fidelity, dp_grid, beta_grid, title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Fidelity contour saved to {}", save_path)
    return fig


# TODO comparision to Gaussian pulse
# OMEGA_GAUSSIAN = 2 * np.pi * 50e3 * 1e-6
# def gaussian_envelope(
#     time_us: np.ndarray, omega_max: float, sigma_us: float
# ) -> np.ndarray:
#     r"""Gaussian pulse envelope.

#     .. math::
#         \Omega_R(t) = \Omega_\max \exp\!\left[-\frac{(t - t_c)^2}{(2\sigma_\tau)^2}\right]

#     Args:
#         time_us: time grid in microseconds
#         omega_max: peak Rabi frequency (rad/us)
#         sigma_us: Gaussian width σ_τ in microseconds

#     Returns:
#         envelope as numpy array
#     """
#     t_center = (time_us[0] + time_us[-1]) / 2
#     return omega_max * np.exp(-((time_us - t_center) ** 2) / (2 * sigma_us) ** 2)

# TODO
# def plot_fidelity_side_by_side(
#     fidelity_opt: np.ndarray,
#     fidelity_gauss: np.ndarray,
#     dp_grid: np.ndarray,
#     beta_grid: np.ndarray,
#     save_path: str | None = None,
# ):
#     """Side-by-side fidelity contour: optimized pulse vs Gaussian pulse."""
#     fig, (ax_opt, ax_gauss) = plt.subplots(1, 2, figsize=(14, 5))
#     _draw_fidelity_contour(ax_opt, fidelity_opt, dp_grid, beta_grid, "Optimized pulse")
#     _draw_fidelity_contour(
#         ax_gauss, fidelity_gauss, dp_grid, beta_grid, "Gaussian pulse"
#     )
#     fig.tight_layout()
#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#         logger.info("Side-by-side fidelity plot saved to {}", save_path)
#     return fig


def _draw_fidelity_contour(ax, fidelity, dp_grid, beta_grid, title):
    """Draw a single fidelity contour panel on the given axes."""
    cf = ax.contourf(
        dp_grid,
        beta_grid,
        fidelity.T,
        levels=np.linspace(0, 1, 51),
        cmap="inferno",
    )
    cs = ax.contour(
        dp_grid,
        beta_grid,
        fidelity.T,
        levels=[0.8, 0.9],
        colors="white",
        linewidths=0.8,
    )
    ax.clabel(cs, fmt="%.2f", fontsize=8)
    ax.figure.colorbar(
        cf,
        ax=ax,
        label="Fidelity",
    )
    ax.set_xlabel(r"$\delta p$ ($\hbar k$)")
    ax.set_ylabel(r"$\beta$ (intensity error)")
    ax.set_title(title)


def generate_report(
    propagator: MirrorPropagator,
    npz_path: str = "mirror_pulse.npz",
    N_grid: int = 41,
    dp_range: tuple = (-0.5, 0.5),
    beta_range: tuple = (-0.2, 0.2),
    output_dir: str | None = None,
):
    """Load saved pulse, evaluate, plot, and log a full report."""
    if output_dir is None:
        output_dir = os.path.dirname(npz_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Derive a prefix from the npz filename (e.g. "20260212_153000_mirror_pulse")
    base = os.path.splitext(os.path.basename(npz_path))[0]
    param_plot_path = os.path.join(output_dir, f"{base}_parameters.png")
    contour_plot_path = os.path.join(output_dir, f"{base}_fidelity_contour.png")

    # TODO
    # gauss_contour_path = os.path.join(output_dir, f"{base}_gaussian_fidelity.png")
    # comparison_path = os.path.join(output_dir, f"{base}_fidelity_comparison.png")

    data = np.load(npz_path)
    device = propagator.device

    Rabi_i = torch.tensor(data["Rabi_i"], dtype=torch.float64, device=device)
    Rabi_q = torch.tensor(data["Rabi_q"], dtype=torch.float64, device=device)
    delta_t = torch.tensor(data["delta"], dtype=torch.float64, device=device)
    time_us = data["time_us"]

    logger.info(
        "Loaded pulse from {} (best_loss={:.6f})", npz_path, float(data["best_loss"])
    )

    # Fidelity evaluation — optimized pulse
    fidelity, dp_grid, beta_grid = evaluate_fidelity_grid(
        propagator,
        Rabi_i,
        Rabi_q,
        delta_t,
        N_grid=N_grid,
        dp_range=dp_range,
        beta_range=beta_range,
    )
    log_fidelity_summary(fidelity)

    # TODO
    # Fidelity evaluation — Gaussian reference pulse
    # gauss_ref = gaussian_envelope(time_us, OMEGA_GAUSSIAN, gauss_sigma_us)
    # gauss_Rabi_i = torch.tensor(gauss_ref, dtype=torch.float64, device=device)
    # gauss_Rabi_q = torch.zeros_like(gauss_Rabi_i)
    # gauss_delta = torch.zeros_like(gauss_Rabi_i)
    # gauss_fidelity, _, _ = evaluate_fidelity_grid(
    #     propagator,
    #     gauss_Rabi_i,
    #     gauss_Rabi_q,
    #     gauss_delta,
    #     N_grid=N_grid,
    #     dp_range=dp_range,
    #     beta_range=beta_range,
    # )
    # logger.info(
    #     "Gaussian pulse (sigma={} us) — mean fidelity: {:.4f}",
    #     gauss_sigma_us,
    #     gauss_fidelity.mean(),
    # )

    # Plots
    plot_parameters(
        data["Rabi_i"],
        data["Rabi_q"],
        data["delta"],
        time_us,
        save_path=param_plot_path,
    )
    plot_fidelity_contour(
        fidelity,
        dp_grid,
        beta_grid,
        title=r"Optimized pulse fidelity $|\operatorname{Tr}(\hat{U}_{\pi}^{\dagger}\hat{U}_{\mathrm{M}})/\operatorname{Tr}(\hat{U}_{\pi}^{\dagger}\hat{U}_{\pi})|^2$",
        save_path=contour_plot_path,
    )
    # TODO
    # plot_fidelity_contour(
    #     gauss_fidelity,
    #     dp_grid,
    #     beta_grid,
    #     title=rf"Gaussian pulse fidelity ($\sigma_\tau={gauss_sigma_us:.0f}\,\mu$s)",
    #     save_path=gauss_contour_path,
    # )
    # TODO
    # plot_fidelity_side_by_side(
    #     fidelity,
    #     gauss_fidelity,
    #     dp_grid,
    #     beta_grid,
    #     save_path=comparison_path,
    # )
