"""
Mirror pulse optimization for order-n Bragg diffraction.
Based on Saywell et al., Nature Communications (2023).

Physics: 87Rb atoms, 2nℏk momentum transfer, Mach-Zehnder interferometer.
Pulse: piecewise-constant steps × 1 μs each.
Cost: unitary infidelity averaged over momentum spread + intensity noise.
Optimizer: Adam with projected constraints (sinc filter, amplitude bounds).
"""

import torch
import numpy as np

from const import amu, hbar, k_L, M_Rb, omega_recoil

from loguru import logger
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info("Using device: {}", device)

# ============================================================
# Pulse and basis parameters
# ============================================================

dt = 1.0  # Hamiltonian propagator time step duration (μs)

N_PULSE_STEPS = 100
NUM_BUFFER_STATES = 4
N_MIRROR = 3  # Bragg order (6ℏk transfer)
OMEGA_MAX = 2 * np.pi * 40e3 * 1e-6  # peak Rabi freq: 2π × 40 kHz → rad/μs

N_STATES = 2 * NUM_BUFFER_STATES + N_MIRROR + 1
m_values = torch.arange(
    -NUM_BUFFER_STATES, N_MIRROR + NUM_BUFFER_STATES + 1, dtype=torch.float64,
    device=device,
)

# Noise distributions
sigma_p = 0.15  # momentum spread (units of ℏk), Gaussian
beta_min, beta_max = -0.15, 0.15  # intensity error, uniform

# Sinc filter cutoff
sinc_cutoff_khz = 80.0  # kHz

# ============================================================
# Helper functions
# ============================================================

def get_u_pi(num_buffer_states, n):
    """Construct the ideal mirror unitary for order-n Bragg diffraction.
    Maps |0⟩ → -i|n⟩ and |n⟩ → -i|0⟩, identity on all other states."""
    size = 2 * num_buffer_states + n + 1
    result = torch.eye(size, dtype=torch.complex128, device=device)
    idx_0 = num_buffer_states
    idx_n = num_buffer_states + n
    result[idx_0, idx_0] = 0
    result[idx_n, idx_n] = 0
    result[idx_0, idx_n] = -1j
    result[idx_n, idx_0] = -1j
    return result


def get_projection(num_buffer_states, n):
    """Projector onto the target 2D subspace {|0⟩, |n⟩}."""
    size = 2 * num_buffer_states + n + 1
    result = torch.zeros(size, size, dtype=torch.complex128, device=device)
    idx_0 = num_buffer_states
    idx_n = num_buffer_states + n
    result[idx_0, idx_0] = 1
    result[idx_n, idx_n] = 1
    return result


def sinc_filter(signal, cutoff_khz, dt_us=1.0):
    """Low-pass filter: zero all Fourier components above cutoff."""
    N = signal.shape[0]
    spectrum = torch.fft.rfft(signal)
    freqs_khz = torch.fft.rfftfreq(N, d=dt_us, device=signal.device).to(signal.dtype) * 1e3
    mask = (freqs_khz.abs() <= cutoff_khz).to(signal.dtype)
    return torch.fft.irfft(spectrum * mask, n=N)


def apply_constraints(Rabi_i, Rabi_q, delta):
    """Project parameters onto feasible set:
    1. Sinc band-limit
    2. Clamp amplitude ≤ OMEGA_MAX
    3. Zero at boundaries
    """
    with torch.no_grad():
        Rabi_i.copy_(sinc_filter(Rabi_i.detach(), sinc_cutoff_khz))
        Rabi_q.copy_(sinc_filter(Rabi_q.detach(), sinc_cutoff_khz))
        delta.copy_(sinc_filter(delta.detach(), sinc_cutoff_khz))

        amp = torch.sqrt(Rabi_i**2 + Rabi_q**2)
        scale = torch.clamp(OMEGA_MAX / (amp + 1e-12), max=1.0)
        Rabi_i.mul_(scale)
        Rabi_q.mul_(scale)

        Rabi_i[0] = 0.0
        Rabi_i[-1] = 0.0
        Rabi_q[0] = 0.0
        Rabi_q[-1] = 0.0


def build_hamiltonians_batched(Rabi_i, Rabi_q, delta, delta_p_batch, beta_batch):
    """
    Build Hamiltonians for all time steps and all noise samples.

    Returns: H of shape (B, T, N, N), complex128
    where B = batch_size, T = N_PULSE_STEPS, N = N_STATES.
    """
    B = delta_p_batch.shape[0]
    T = Rabi_i.shape[0]
    N = N_STATES

    # Derive Rabi amplitude and phase from quadratures
    omega_R = torch.sqrt(Rabi_i**2 + Rabi_q**2)  # (T,)
    phi_L = torch.atan2(Rabi_q, Rabi_i)  # (T,)

    # --- Diagonal entries: ω_R × (2m + δ_p + δ/(4ω_R))² ---
    dp = delta_p_batch[:, None, None]  # (B,1,1)
    dd = delta[None, :, None]  # (1,T,1)
    mv = m_values[None, None, :]  # (1,1,N)

    arg = 2 * mv + dp + dd / (4 * omega_recoil)  # (B,T,N)
    diag_vals = omega_recoil * arg**2  # (B,T,N)

    # --- Off-diagonal entries: (1+β)Ω_R × exp(±iφ_L/2) ---
    omega_eff = (1 + beta_batch[:, None]) * omega_R[None, :]  # (B,T)
    phase_factor = torch.exp(-1j * phi_L.to(torch.complex128) / 2)  # (T,)

    # --- Assemble tridiagonal H ---
    H = torch.zeros(B, T, N, N, dtype=torch.complex128, device=Rabi_i.device)

    idx = torch.arange(N, device=Rabi_i.device)
    H[:, :, idx, idx] = diag_vals.to(torch.complex128)

    idx_off = torch.arange(N - 1, device=Rabi_i.device)
    H[:, :, idx_off, idx_off + 1] = (
        omega_eff.to(torch.complex128)[:, :, None] * phase_factor[None, :, None]
    ).expand(B, T, N - 1)
    H[:, :, idx_off + 1, idx_off] = (
        omega_eff.to(torch.complex128)[:, :, None] * phase_factor[None, :, None].conj()
    ).expand(B, T, N - 1)

    return H


def compute_propagator_batched(H_all):
    """Chain-multiply matrix exponentials over time.
    H_all: (B, T, N, N) → U_total: (B, N, N)."""
    B, T, N, _ = H_all.shape
    U_steps = torch.linalg.matrix_exp(-1j * H_all * dt)
    U_total = (
        torch.eye(N, dtype=torch.complex128, device=H_all.device)
        .unsqueeze(0).expand(B, -1, -1).clone()
    )
    for t in range(T):
        U_total = U_steps[:, t] @ U_total
    return U_total


def mirror_cost_batched(U_total, U_pi, projection):
    """
    Compute mirror infidelity for a batch of propagators.
    F = |Tr(P @ U_π† @ U)|² / |Tr(P @ U_π† @ U_π)|²
    U_total: (B, N, N), returns cost of shape (B,).
    """
    M = projection @ U_pi.conj().T  # (N, N)
    overlap = torch.einsum("ij,bij->b", M, U_total)  # (B,)
    norm = torch.trace(M @ U_pi).real  # scalar = 2
    cost = 1.0 - (overlap.abs() / norm) ** 2
    return cost

# Optimization hyperparameters
batch_size = 128
num_epochs = 1
lr = 0.001

# ============================================================
# Initialization
# ============================================================

U_pi = get_u_pi(NUM_BUFFER_STATES, N_MIRROR)
projection = get_projection(NUM_BUFFER_STATES, N_MIRROR)

# Optimization variables: in-phase/quadrature Rabi components + detuning
Rabi_i = torch.randn(N_PULSE_STEPS, dtype=torch.float64, device=device) * OMEGA_MAX * 0.3
Rabi_q = torch.randn(N_PULSE_STEPS, dtype=torch.float64, device=device) * OMEGA_MAX * 0.3
delta = torch.randn(N_PULSE_STEPS, dtype=torch.float64, device=device) * omega_recoil * 0.1

Rabi_i.requires_grad_(True)
Rabi_q.requires_grad_(True)
delta.requires_grad_(True)

apply_constraints(Rabi_i, Rabi_q, delta)

optimizer = torch.optim.Adam([Rabi_i, Rabi_q, delta], lr=lr)

# ============================================================
# Training loop
# ============================================================

best_loss = float("inf")
best_params = None

for iteration in tqdm(range(num_epochs), desc="Optimizing"):
    optimizer.zero_grad()

    # Draw fresh noise samples
    delta_p_samples = torch.randn(batch_size, dtype=torch.float64, device=device) * sigma_p
    beta_samples = (
        torch.rand(batch_size, dtype=torch.float64, device=device)
        * (beta_max - beta_min) + beta_min
    )

    # Forward pass
    H_all = build_hamiltonians_batched(
        Rabi_i, Rabi_q, delta, delta_p_samples, beta_samples
    )
    U_total = compute_propagator_batched(H_all)
    costs = mirror_cost_batched(U_total, U_pi, projection)
    loss = costs.mean()

    # Backward pass + optimizer step
    loss.backward()
    optimizer.step()
    apply_constraints(Rabi_i, Rabi_q, delta)

    # Logging
    loss_val = loss.item()
    if loss_val < best_loss:
        best_loss = loss_val
        best_params = (
            Rabi_i.detach().clone(),
            Rabi_q.detach().clone(),
            delta.detach().clone(),
        )

    if iteration % 100 == 0 or iteration == num_epochs - 1:
        omega_R = torch.sqrt(Rabi_i**2 + Rabi_q**2).detach()
        logger.info(
            "Iter {:5d} | loss: {:.6f} | best: {:.6f} | peak Omega_R: {:.4f} rad/us",
            iteration, loss_val, best_loss, omega_R.max().item(),
        )

# ============================================================
# Save results
# ============================================================

Rabi_i_best, Rabi_q_best, delta_best = best_params
omega_R_best = torch.sqrt(Rabi_i_best**2 + Rabi_q_best**2).cpu().numpy()
phi_L_best = torch.atan2(Rabi_q_best, Rabi_i_best).cpu().numpy()

np.savez(
    "mirror_pulse.npz",
    omega_R=omega_R_best,
    phi_L=phi_L_best,
    delta=delta_best.cpu().numpy(),
    Rabi_i=Rabi_i_best.cpu().numpy(),
    Rabi_q=Rabi_q_best.cpu().numpy(),
    time_us=np.arange(N_PULSE_STEPS) * dt,
    best_loss=best_loss,
)

logger.info("Optimization complete. Best loss: {:.6f}", best_loss)
logger.info("Saved to mirror_pulse.npz")

# ============================================================
# Evaluation: test on a fine grid of (δ_p, β)
# ============================================================

logger.info("Evaluating on a 21x21 grid of (delta_p, beta)...")
with torch.no_grad():
    dp_grid = torch.linspace(-0.5, 0.5, 21, dtype=torch.float64, device=device)
    beta_grid = torch.linspace(-0.2, 0.2, 21, dtype=torch.float64, device=device)
    dp_mesh, beta_mesh = torch.meshgrid(dp_grid, beta_grid, indexing="ij")

    H_eval = build_hamiltonians_batched(
        Rabi_i_best, Rabi_q_best, delta_best, dp_mesh.flatten(), beta_mesh.flatten()
    )
    U_eval = compute_propagator_batched(H_eval)
    costs_eval = mirror_cost_batched(U_eval, U_pi, projection)

    fidelity = 1 - costs_eval.reshape(21, 21).cpu().numpy()

    logger.info("Mean fidelity over grid: {:.4f}", fidelity.mean())
    logger.info("Min fidelity over grid:  {:.4f}", fidelity.min())
    logger.info("Fidelity at (delta_p=0, beta=0): {:.6f}", fidelity[10, 10])
    logger.info("Fidelity > 0.9 fraction:  {:.2f}", (fidelity > 0.9).mean())
