# Quantum Cymatics – Resonance Geometry Simulations

This repository contains numerical experiments related to the paper:

> "Quantum Cymatics: Revealing Electron Resonance Geometry via Vacuum–Acoustic Coherence Mapping"

The code provides simple 3D simulations of model resonance geometries:

- Toroidal resonance
- Helical resonance
- DLSFH-like (dodecahedral) multi-shell resonance

and computes basic observables:
- Energy density
- Current density
- Far-field angular pattern (via 3D FFT)

## Installation

```bash
git clone <your_repo_url>.git
cd quantum-cymatics-sim
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running Simulations

Basic usage:

```bash
python main.py --geometry torus
python main.py --geometry helix
python main.py --geometry dlsfh
```

Options:

```bash
python main.py --geometry torus --N 64 --R0 0.7 --sigma 0.15 --m 1
python main.py --geometry helix --N 64 --R0 0.6 --sigma 0.12 --m 1 --kz 2.0
python main.py --geometry dlsfh --N 64 --r0 0.7 --sigma 0.1
```

The script will:
- Generate a 3D complex field `psi(x,y,z)` for the chosen geometry
- Compute energy and current densities
- Compute a far-field pattern using a 3D FFT
- Save plots in `./output/` as PNG files

## Files

- `grid.py`: 3D grid creation (dimensionless coordinates scaled to λ̄_C = 1)
- `geometries.py`: model resonance fields (toroidal, helical, DLSFH-like)
- `observables.py`: energy density, current density, far-field angular map
- `main.py`: command-line driver to run simulations and produce plots

This code is a numerical toy model consistent with the conceptual framework of the paper, not a full physical SGCV–MC solver.
