import os
from datetime import datetime, timezone

import torch
import numpy as np

from loguru import logger


def get_cpu_gpu(cuda_number=0):
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        dev_count = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {dev_count}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA:{i} Device name: {torch.cuda.get_device_name(i)}")

    logger.info(f"MPS available: {torch.backends.mps.is_available()}")

    device = torch.device(
        f"cuda:{cuda_number}"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Current device: {device}")

    return device


def save_pulse(
    best_params,
    best_loss: float,
    n_pulse_steps: int,
    dt_pulse: float,
    output_dir: str = "outputs",
):
    """Save optimized pulse parameters to a timestamped .npz file."""
    Rabi_i_best, Rabi_q_best, delta_best = best_params
    omega_R_best = torch.sqrt(Rabi_i_best**2 + Rabi_q_best**2).cpu().numpy()
    phi_L_best = torch.atan2(Rabi_q_best, Rabi_i_best).cpu().numpy()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "mirror_pulse.npz")

    np.savez(
        path,
        omega_R=omega_R_best,
        phi_L=phi_L_best,
        delta=delta_best.cpu().numpy(),
        Rabi_i=Rabi_i_best.cpu().numpy(),
        Rabi_q=Rabi_q_best.cpu().numpy(),
        time_us=np.arange(n_pulse_steps) * dt_pulse,
        best_loss=best_loss,
    )

    logger.info("Saved to {}", path)
    return path
