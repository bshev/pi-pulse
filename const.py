import numpy as np

# ============================================================
# Physical constants
# ============================================================
amu = 1.66053906892e-27
hbar = 1.054571817e-34  # J·s
k_L = 2 * np.pi / 780.241e-9  # laser wavenumber (m^-1), Rb-87 D2 line (780.241 nm)
M_Rb = 86.90918053 * amu  # Rb-87 mass (kg)

# Single-photon recoil frequency (rad/s), converted to rad/μs
omega_recoil = hbar * k_L**2 / (2 * M_Rb) * 1e-6  # ≈ 0.0237 rad/μs
