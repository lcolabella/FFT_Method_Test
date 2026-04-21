# Brinkman Penalization Solver (Option A)

## Summary
The Brinkman solver computes flow on the full voxel grid and suppresses velocity in solid voxels with a large penalization term.

Governing equations:

$$
-\nabla p + \mu \nabla^2 \mathbf{u} - \alpha(\mathbf{x})\,\mathbf{u} = \mathbf{f}, \qquad \nabla\cdot\mathbf{u}=0
$$

where:
- $\alpha(\mathbf{x}) = 0$ in fluid voxels
- $\alpha(\mathbf{x}) = \alpha_s \gg 1$ in solid voxels
- $\mathbf{f}$ is an imposed body-force equivalent to macroscopic pressure gradient

## Current Implementation Notes
- Implemented in [src/FluidBrinkman.cpp](../src/FluidBrinkman.cpp).
- Supports two run modes:
  - permeability: computes full 3x3 permeability tensor
  - response: computes velocity field for a user gradient
- Supports phase-tag based fluid/solid selection from material file.

## Strengths
- Robust baseline solver.
- Simple parameterization and stable behavior for small/medium cases.
- Natural fit for voxelized FFT workflows.

## Limitations
- Penalization parameter tuning can influence conditioning.
- Interface accuracy can degrade if penalty is not well calibrated.

## Recommended Usage
Use this solver when you need a stable reference baseline:
- fluid_solver = brinkman

## References
1. Angot, Bruneau, Fabrie (1999). A penalization method to take into account obstacles in incompressible viscous flows. Numerische Mathematik 81, 497-520.
2. Sanchez-Palencia (1980). Non-Homogeneous Media and Vibration Theory.
3. Darcy (1856). Les fontaines publiques de la ville de Dijon.
