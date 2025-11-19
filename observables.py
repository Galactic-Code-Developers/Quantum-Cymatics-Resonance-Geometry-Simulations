import numpy as np
from numpy.fft import fftn, fftshift


def energy_density(psi, dx, V_eff=0.0):
    """Compute a simple energy-density functional:

    E(x) = |∇ψ|^2 + V_eff * |ψ|^2

    where derivatives are computed via central finite differences.

    Parameters
    ----------
    psi : ndarray (complex)
        Wave-like resonance field.
    dx : float
        Grid spacing.
    V_eff : float or ndarray
        Effective potential term (toy model).

    Returns
    -------
    E : ndarray (float)
        Energy density on the grid.
    """
    grad2 = 0.0
    for axis in range(3):
        psi_fwd = np.roll(psi, -1, axis=axis)
        psi_bwd = np.roll(psi, +1, axis=axis)
        dpsi = (psi_fwd - psi_bwd) / (2.0 * dx)
        grad2 += np.abs(dpsi)**2

    E = grad2 + V_eff * np.abs(psi)**2
    return E


def current_density(psi, dx):
    """Compute a simple probability / resonance current density:

    J = Im(ψ* ∇ψ)

    Parameters
    ----------
    psi : ndarray (complex)
        Wave-like resonance field.
    dx : float
        Grid spacing.

    Returns
    -------
    Jx, Jy, Jz : ndarray
        Components of the current density.
    """
    J = []
    for axis in range(3):
        psi_fwd = np.roll(psi, -1, axis=axis)
        dpsi = (psi_fwd - psi) / dx
        comp = np.imag(np.conj(psi) * dpsi)
        J.append(comp)

    Jx, Jy, Jz = J
    return Jx, Jy, Jz


def farfield_pattern(psi):
    """Compute a far-field angular pattern using a 3D FFT.

    The far-field amplitude A(k) ≈ FFT[ψ(x)],
    and intensity ∝ |A(k)|^2. For simplicity we return the
    3D intensity in k-space and a collapsed 2D slice.

    Parameters
    ----------
    psi : ndarray (complex)
        Wave-like resonance field.

    Returns
    -------
    I3D : ndarray (float)
        3D intensity in k-space (centered).
    I2D : ndarray (float)
        2D central slice (kz=0) of the intensity.
    """
    A = fftshift(fftn(psi))
    I3D = np.abs(A)**2

    kz_mid = I3D.shape[2] // 2
    I2D = I3D[:, :, kz_mid]

    if I3D.max() > 0:
        I3D /= I3D.max()
    if I2D.max() > 0:
        I2D /= I2D.max()

    return I3D, I2D
