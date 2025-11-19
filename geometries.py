import numpy as np

# --- basic constants used in toy models ---
# these are dimensionless (Compton-scaled) in this code
DEFAULT_A0 = 1.0


def toroidal_resonance(x, y, z, R0=0.7, sigma=0.15, m=1, A0=DEFAULT_A0):
    """Toroidal resonance geometry (Appendix C.2).

    ψ_tor(ρ, φ, z) = A0 * exp( -[(sqrt(ρ^2+z^2) - R0)^2] / (2σ^2) ) * exp(i m φ)

    Parameters
    ----------
    x, y, z : ndarray
        3D coordinate grids.
    R0 : float
        Dimensionless torus radius.
    sigma : float
        Cross-sectional thickness.
    m : int
        Azimuthal winding number.
    A0 : float
        Overall amplitude.

    Returns
    -------
    psi : ndarray (complex)
        3D complex field.
    """
    rho = np.sqrt(x**2 + y**2)
    R = np.sqrt(rho**2 + z**2)
    phi = np.arctan2(y, x)
    envelope = np.exp(-((R - R0) ** 2) / (2.0 * sigma**2))
    phase = np.exp(1j * m * phi)
    psi = A0 * envelope * phase
    return psi


def helical_resonance(x, y, z, R0=0.6, sigma=0.12, m=1, kz=2.0, A1=DEFAULT_A0):
    """Helical phase-winding resonance (Appendix C.3).

    ψ_helix(ρ, φ, z) = A1 * exp( -[(ρ - R0)^2 + z^2] / (2σ^2) )
                       * exp(i (m φ + k_z z))

    Parameters
    ----------
    x, y, z : ndarray
        3D coordinate grids.
    R0 : float
        Ring radius in the x-y plane.
    sigma : float
        Transverse Gaussian width.
    m : int
        Angular winding number.
    kz : float
        Longitudinal phase ramp along z.
    A1 : float
        Amplitude.

    Returns
    -------
    psi : ndarray (complex)
        3D complex field.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    envelope = np.exp(-(((rho - R0) ** 2) + z**2) / (2.0 * sigma**2))
    phase = np.exp(1j * (m * phi + kz * z))
    psi = A1 * envelope * phase
    return psi


def dlsfh_resonance(x, y, z, r0=0.7, sigma=0.1, A=DEFAULT_A0):
    """Dodecahedral-like (DLSFH-type) multi-shell resonance (Appendix C.4).

    ψ_DLSFH(x) = Σ_k A_k exp( -|x - r0 n_k|^2 / (2σ_k^2) ) e^{i φ_k}

    Here we use fixed amplitudes and phases for demonstration, and a set
    of approximate dodecahedron vertex directions.

    Parameters
    ----------
    x, y, z : ndarray
        3D coordinate grids.
    r0 : float
        Shell radius in Compton units.
    sigma : float
        Gaussian width around each vertex.
    A : float
        Overall amplitude scaling.

    Returns
    -------
    psi : ndarray (complex)
        3D complex field with localized lobes.
    """
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0

    # Build a simple approximate set of dodecahedron-like directions.
    base = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            base.append((0.0, s1 / phi_g, s2 * phi_g))
            base.append((s1 / phi_g, s2 * phi_g, 0.0))
            base.append((s1 * phi_g, 0.0, s2 / phi_g))

    # Remove duplicates and normalize
    verts_unique = []
    for v in base:
        if v not in verts_unique:
            verts_unique.append(v)

    directions = []
    for (vx, vy, vz) in verts_unique:
        v = np.array([vx, vy, vz], dtype=float)
        norm = np.linalg.norm(v)
        if norm > 0:
            directions.append(v / norm)

    psi = np.zeros_like(x, dtype=complex)

    Ndir = len(directions)
    for k, n_vec in enumerate(directions):
        nx, ny, nz = n_vec
        cx = r0 * nx
        cy = r0 * ny
        cz = r0 * nz
        r2 = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
        phi_k = 2.0 * np.pi * k / max(Ndir, 1)
        psi += np.exp(-r2 / (2.0 * sigma**2)) * np.exp(1j * phi_k)

    psi *= A
    return psi
