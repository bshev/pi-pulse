"""
Mirror pulse optimization for order-n Bragg diffraction.
Based on Saywell et al., Nature Communications (2023).
"""

import torch
import numpy as np

from utils import get_cpu_gpu, save_pulse
from physics import MirrorPropagator, apply_constraints
from evaluation import generate_report

from const import omega_recoil

from loguru import logger
from tqdm import tqdm

# Will probably be very slow without gpu
# Does not support MPS (float64)
device = get_cpu_gpu()

# ============================================================
# Pulse control sequence and hamiltonian basis parameters
# ============================================================

N_MIRROR = 3  # Bragg order (6ℏk transfer)
NUM_BUFFER_STATES = 4 #number of states to include in hamiltonian matrix on either side of |p> and |p+6>

N_PULSE_STEPS = 200
DT_PULSE = 1  # pulse time step units in us
OMEGA_MAX = 2 * np.pi * 40e3 * 1e-6  # peak Rabi freq: 2π × 40 kHz → rad/μs
SINC_CUTOFF_KHZ = 80.0  # kHz

# Noise distributions
sigma_p = 0.15  # momentum spread (units of ℏk), Gaussian
beta_min, beta_max = -0.15, 0.15  # intensity error, uniform

# Optimization loop hyperparameters
batch_size = 256
num_epochs = 1
lr = 0.001  # optimizer learning rate

# set to .npz path to resume, e.g. "outputs/20260212_182306/mirror_pulse.npz"
LOAD_FILE = None
#LOAD_FILE = "/home/bshev/Dev/pi-pulse/outputs/20260212_184034/mirror_pulse.npz"  

pi_pulse = MirrorPropagator(N_MIRROR, NUM_BUFFER_STATES, DT_PULSE, device)

# ============================================================
# Initialization
# ============================================================
if LOAD_FILE is not None:
    data = np.load(LOAD_FILE)
    Rabi_i = torch.tensor(data["Rabi_i"], dtype=torch.float64, device=device)
    Rabi_q = torch.tensor(data["Rabi_q"], dtype=torch.float64, device=device)
    delta = torch.tensor(data["delta"], dtype=torch.float64, device=device)
    logger.info(
        "Loaded pulse from {} (loss={:.6f})", LOAD_FILE, float(data["best_loss"])
    )
else:
    # Optimization variables: in-phase/quadrature Rabi components + detuning
    # can recover amp/phase if needed.
    Rabi_i = (
        torch.randn(N_PULSE_STEPS, dtype=torch.float64, device=device) * OMEGA_MAX * 0.3
    )
    Rabi_q = (
        torch.randn(N_PULSE_STEPS, dtype=torch.float64, device=device) * OMEGA_MAX * 0.3
    )
    delta = (
        torch.randn(N_PULSE_STEPS, dtype=torch.float64, device=device)
        * omega_recoil
        * 0.1
    )

Rabi_i.requires_grad_(True)
Rabi_q.requires_grad_(True)
delta.requires_grad_(True)

apply_constraints(Rabi_i, Rabi_q, delta, OMEGA_MAX, SINC_CUTOFF_KHZ, device)

optimizer = torch.optim.Adam([Rabi_i, Rabi_q, delta], lr=lr)

# ============================================================
# Training loop
# ============================================================

best_loss = float("inf")
best_params = None

for batch in tqdm(range(num_epochs), desc="Optimizing pulse paramters"):
    optimizer.zero_grad()

    # Draw batch of samples from noise distribution
    delta_p_samples = (
        torch.randn(batch_size, dtype=torch.float64, device=device) * sigma_p
    )
    beta_samples = (
        torch.rand(batch_size, dtype=torch.float64, device=device)
        * (beta_max - beta_min)
        + beta_min
    )

    # Forward pass
    H_all = pi_pulse.build_hamiltonians(
        Rabi_i, Rabi_q, delta, delta_p_samples, beta_samples
    )
    U_total = pi_pulse.compute_propagator(H_all)
    loss = pi_pulse.mirror_loss(U_total)

    # Backward pass + optimizer step + constrained optimization
    loss.backward()
    optimizer.step()
    apply_constraints(Rabi_i, Rabi_q, delta, OMEGA_MAX, SINC_CUTOFF_KHZ, device)

    # Logging
    loss_val = loss.item()
    if loss_val < best_loss:
        best_loss = loss_val
        best_params = (
            Rabi_i.detach().clone(),
            Rabi_q.detach().clone(),
            delta.detach().clone(),
        )

    if batch % 100 == 0 or batch == num_epochs - 1:
        omega_R = torch.sqrt(Rabi_i**2 + Rabi_q**2).detach()
        logger.info(
            "Iter {:5d} | loss: {:.6f} | best: {:.6f} | peak Omega_R: {:.4f} rad/us",
            batch,
            loss_val,
            best_loss,
            omega_R.max().item(),
        )

# ============================================================
# Save results
# ============================================================

logger.info("Optimization complete. Best loss: {:.6f}", best_loss)
pulse_path = save_pulse(best_params, best_loss, N_PULSE_STEPS, DT_PULSE)

# Free training state before evaluation
del Rabi_i, Rabi_q, delta, optimizer, best_params
torch.cuda.empty_cache()

generate_report(pi_pulse, npz_path=pulse_path)
