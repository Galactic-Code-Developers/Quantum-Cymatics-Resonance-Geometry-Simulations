import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from grid import create_grid
from geometries import toroidal_resonance, helical_resonance, dlsfh_resonance
from observables import energy_density, current_density, farfield_pattern


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum Cymatics numerical experiments"
    )
    parser.add_argument(
        "--geometry",
        type=str,
        choices=["torus", "helix", "dlsfh"],
        default="torus",
        help="Resonance geometry to simulate.",
    )
    parser.add_argument(
        "--N", type=int, default=64, help="Grid size in each dimension."
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=2.0,
        help="Half-extent of the dimensionless domain.",
    )

    parser.add_argument("--R0", type=float, default=0.7, help="Torus / helix radius.")
    parser.add_argument("--sigma", type=float, default=0.15, help="Gaussian width.")
    parser.add_argument("--m", type=int, default=1, help="Azimuthal winding number.")
    parser.add_argument("--kz", type=float, default=2.0, help="Helical k_z.")
    parser.add_argument("--r0", type=float, default=0.7, help="DLSFH shell radius.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save plots.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    x, y, z, dx = create_grid(N=args.N, extent=args.extent)

    if args.geometry == "torus":
        psi = toroidal_resonance(
            x, y, z, R0=args.R0, sigma=args.sigma, m=args.m
        )
        tag = "torus"
    elif args.geometry == "helix":
        psi = helical_resonance(
            x, y, z, R0=args.R0, sigma=args.sigma, m=args.m, kz=args.kz
        )
        tag = "helix"
    else:
        psi = dlsfh_resonance(
            x, y, z, r0=args.r0, sigma=args.sigma
        )
        tag = "dlsfh"

    E = energy_density(psi, dx)
    Jx, Jy, Jz = current_density(psi, dx)
    I3D, I2D = farfield_pattern(psi)

    total_energy = np.sum(E) * dx**3
    norm = np.sum(np.abs(psi) ** 2) * dx**3

    print(f"[{tag}] Grid: {args.N}^3, dx={dx:.3e}")
    print(f"[{tag}] Norm(psi) ≈ {norm:.6e}")
    print(f"[{tag}] Total energy (toy functional) ≈ {total_energy:.6e}")

    mid = args.N // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(
        np.abs(psi[:, :, mid]) ** 2,
        origin="lower",
        extent=[-args.extent, args.extent, -args.extent, args.extent],
    )
    axes[0].set_title("|ψ|² (z=0 slice)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        np.abs(psi[:, mid, :]) ** 2,
        origin="lower",
        extent=[-args.extent, args.extent, -args.extent, args.extent],
    )
    axes[1].set_title("|ψ|² (y=0 slice)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        np.abs(psi[mid, :, :]) ** 2,
        origin="lower",
        extent=[-args.extent, args.extent, -args.extent, args.extent],
    )
    axes[2].set_title("|ψ|² (x=0 slice)")
    plt.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f"{tag}_psi_slices.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    imE = ax.imshow(
        E[:, :, mid],
        origin="lower",
        extent=[-args.extent, args.extent, -args.extent, args.extent],
    )
    ax.set_title("Energy density (z=0 slice)")
    plt.colorbar(imE, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f"{tag}_energy_slice.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    imF = ax.imshow(
        I2D,
        origin="lower",
        extent=[-1, 1, -1, 1],
    )
    ax.set_title("Far-field pattern (kz=0 slice, normalized)")
    plt.colorbar(imF, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f"{tag}_farfield_slice.png"), dpi=200)
    plt.close(fig)

    print(f"[{tag}] Plots saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
