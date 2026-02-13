import torch

from const import omega_recoil


def sinc_filter(signal, cutoff_khz, device, dt_us=1.0):
    """Low-pass filter: zero all Fourier components above cutoff.
    cutoff in khz, time steps in us
    """
    N = signal.shape[0]
    spectrum = torch.fft.rfft(signal)  # control sequence FFT
    freqs_khz = (
        torch.fft.rfftfreq(N, d=dt_us, device=signal.device).to(signal.dtype) * 1e3
    )  # FFT frequencies
    mask = (freqs_khz <= cutoff_khz).to(signal.dtype)
    return torch.fft.irfft(
        spectrum * mask, n=N
    )  # filtering and recovering control sequence


def apply_constraints(
    Rabi_i, Rabi_q, delta, omega_max, sinc_cutoff_khz, device, eps=1e-12
):
    """Project parameters onto feasible set:
    1. Zero at boundaries (pre-filter, so sinc sees the zeros)
    2. Sinc band-limit
    3. Clamp amplitude ≤ OMEGA_MAX
    4. Re-zero boundaries + eps offset on Rabi_i to keep atan2 safe
    """
    with torch.no_grad():
        Rabi_i[0] = eps  # without small numerical offset get nans
        Rabi_i[-1] = eps
        Rabi_q[0] = eps
        Rabi_q[-1] = eps

        Rabi_i.copy_(sinc_filter(Rabi_i.detach(), sinc_cutoff_khz, device))
        Rabi_q.copy_(sinc_filter(Rabi_q.detach(), sinc_cutoff_khz, device))
        delta.copy_(sinc_filter(delta.detach(), sinc_cutoff_khz, device))

        amp = torch.sqrt(Rabi_i**2 + Rabi_q**2)
        scale = torch.clamp(omega_max / (amp + 1e-12), max=1.0)
        Rabi_i.mul_(scale)
        Rabi_q.mul_(scale)

        Rabi_i[0] = eps
        Rabi_i[-1] = eps
        Rabi_q[0] = eps
        Rabi_q[-1] = eps


class MirrorPropagator:
    """Encapsulates Bragg diffraction mirror propagator construction and loss computation.

    Caches U_pi and the projection operator on construction so they are
    built exactly once for a given (n_mirror, num_buffer_states) pair.
    """

    def __init__(self, n_mirror, num_buffer_states, dt_pulse, device):
        self.n_mirror = n_mirror
        self.num_buffer_states = num_buffer_states
        self.dt_pulse = dt_pulse
        self.device = device
        self.n_states = 2 * num_buffer_states + n_mirror + 1

        # Cached matrices
        self.U_pi = self._build_u_pi()
        self.projection = self._build_projection()

        # Precompute momentum-state index vector
        self.m_values = torch.arange(
            -num_buffer_states,
            n_mirror + num_buffer_states + 1,
            dtype=torch.float64,
            device=device,
        )

    def _build_u_pi(self):
        """Construct the ideal mirror unitary for order-n Bragg diffraction.
        Maps |0> -> -i|n> and |n> -> -i|0>, identity on all other states."""
        N = self.n_states
        result = torch.eye(N, dtype=torch.complex128, device=self.device)
        idx_0 = self.num_buffer_states
        idx_n = self.num_buffer_states + self.n_mirror
        result[idx_0, idx_0] = 0
        result[idx_n, idx_n] = 0
        result[idx_0, idx_n] = -1j
        result[idx_n, idx_0] = -1j
        return result

    def _build_projection(self):
        """Projector onto the target 2D subspace {|0>, |n>}."""
        N = self.n_states
        result = torch.zeros(N, N, dtype=torch.complex128, device=self.device)
        idx_0 = self.num_buffer_states
        idx_n = self.num_buffer_states + self.n_mirror
        result[idx_0, idx_0] = 1
        result[idx_n, idx_n] = 1
        return result

    def build_hamiltonians(self, Rabi_i, Rabi_q, delta, delta_p_batch, beta_batch):
        """
        Build Hamiltonians for all time steps and all noise samples.

        Returns: H of shape (B, T, N, N), complex128
        where B = batch_size, T = N_PULSE_STEPS, N = N_STATES.
        """
        B = delta_p_batch.shape[0]
        T = Rabi_i.shape[0]
        N = self.n_states

        # Derive Rabi amplitude and phase from i/q
        omega_R = torch.sqrt(Rabi_i**2 + Rabi_q**2)  # (T,)
        phi_L = torch.atan2(Rabi_q, Rabi_i) % (2 * torch.pi)  # branch cuts

        # --- Diagonal entries: omega_R x (2m + delta_p + delta/(4*omega_recoil))^2 ---
        dp = delta_p_batch[:, None, None]  # (B,1,1)
        dd = delta[None, :, None]  # (1,T,1)
        mv = self.m_values[None, None, :]  # (1,1,N)

        arg = 2 * mv + dp + dd / (4 * omega_recoil)  # (B,T,N)
        diag_vals = omega_recoil * arg**2  # (B,T,N)

        # --- Off-diagonal entries: (1+beta)*Omega_R * exp(+-i*phi_L/2) ---
        omega_eff = (1 + beta_batch[:, None]) * omega_R[None, :]  # (B,T)
        phase_factor = torch.exp(-1j * phi_L.to(torch.complex128) / 2)  # (T,)

        # --- Assemble tridiagonal H ---
        H = torch.zeros(B, T, N, N, dtype=torch.complex128, device=self.device)

        idx = torch.arange(N, device=self.device)
        H[:, :, idx, idx] = diag_vals.to(torch.complex128)

        idx_off = torch.arange(N - 1, device=self.device)
        H[:, :, idx_off, idx_off + 1] = (
            omega_eff.to(torch.complex128)[:, :, None] * phase_factor[None, :, None]
        ).expand(B, T, N - 1)
        H[:, :, idx_off + 1, idx_off] = (
            omega_eff.to(torch.complex128)[:, :, None]
            * phase_factor[None, :, None].conj()
        ).expand(B, T, N - 1)

        return H

    def compute_propagator(self, H_all):
        """Chain-multiply matrix exponentials over time.
        H_all: (B, T, N, N) -> U_total: (B, N, N)."""
        B, T, N, _ = H_all.shape
        U_steps = torch.linalg.matrix_exp(-1j * H_all * self.dt_pulse)
        U_total = (
            torch.eye(N, dtype=torch.complex128, device=self.device)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clone()
        )
        for t in range(T):
            U_total = U_steps[:, t] @ U_total
        return U_total

    def mirror_loss(self, U_total, reduction="mean"):
        """
        Compute mirror infidelity for a batch of propagators.
        F = |Tr(P @ U_pi_dag @ U)|^2 / |Tr(P @ U_pi_dag @ U_pi)|^2
        U_total: (B, N, N), returns scalar or shape (B,) depending on reduction.
        """
        M = self.projection @ self.U_pi.conj().T  # (N, N)
        overlap = torch.einsum("ij,bij->b", M, U_total)  # (B,)
        norm = torch.trace(M @ self.U_pi).real  # scalar = 2
        loss = 1.0 - (overlap.abs() / norm) ** 2
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
