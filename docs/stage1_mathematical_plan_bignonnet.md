# Stage 1: Mathematical Implementation Plan for FFT-Based Permeability Solver
## Based on François Bignonnet (2020), *Efficient FFT-based upscaling of the permeability of porous media discretized on uniform grids with estimation of RVE size*

**Author:** François Bignonnet  
**Journal:** Computer Methods in Applied Mechanics and Engineering, Volume 369 (2020) 113237  
**DOI:** 10.1016/j.cma.2020.113237  
**Status:** Stage 1 only - no code implementation yet

---

## 1. Macroscopic Problem: Darcy's Law

At the macroscopic scale, incompressible Stokes flow through porous media is governed by Darcy's law:

$$
\begin{align}
\nabla \cdot V &= 0 \label{eq:darcy_div}\\
V &= -K \cdot \nabla P \label{eq:darcy_law}
\end{align}
$$

where:
- $V$ is the macroscopic (homogenized) filtration velocity
- $K$ is the homogenized permeability tensor (the target quantity)
- $\nabla P$ is the imposed macroscopic pressure gradient
- *Paper reference: Eq. (1)*

---

## 2. Microscopic Problem: Periodic Homogenization of Stokes Flow

The microscopic problem is defined on a Representative Volume Element (RVE) $\Omega$ containing a fluid phase $\Omega_f$, solid phase $\Omega_s$, and interface $\Gamma$:

$$
\begin{align}
\nabla \cdot \boldsymbol{\sigma} &= \nabla P \quad \text{in }(\Omega_f) \label{eq:stokes_div}\\
\boldsymbol{\sigma} &= -p\mathbf{1} + 2 \mu \nabla^s \mathbf{v} \quad \text{in }(\Omega_f) \label{eq:newtonian}\\
\nabla \cdot \mathbf{v} &= 0 \quad \text{in }(\Omega_f) \label{eq:stokes_incomp}\\
\mathbf{v} &= 0 \quad \text{on }(\Gamma) \label{eq:noslip}\\
\mathbf{v}, p \text{ periodic} \quad &\text{on }(\partial\Omega) \label{eq:periodic}
\end{align}
$$

where:
- $\mathbf{v}$ is the microscopic fluid velocity (extended as zero in $\Omega_s$)
- $p$ is the microscopic pressure fluctuation
- $\boldsymbol{\sigma}$ is the Cauchy stress tensor
- $\mu$ is the fluid viscosity
- $\nabla P$ is the imposed macroscopic pressure gradient
- $\nabla^sv = \frac{1}{2}(\nabla v + \nabla v^T)$ is the symmetric gradient of $v$
- *Paper reference: Eq. (2)*

The macroscopic velocity is the volume average of the microscopic velocity:

$$
V = \langle \mathbf{v} \rangle = \frac{1}{|\Omega|} \int_{\Omega} \mathbf{v} \, dV
$$

where $\mathbf{v}$ is zero in the solid phase.  
*Paper reference: Eq. (3)*

---

## 3. Auxiliary Problem and Variational Framework

### 3.1 Auxiliary Problem

An auxiliary problem is formulated in which the entire RVE is filled with a Newtonian fluid of uniform viscosity $\mu$, with heterogeneity and the no-slip condition accounted for indirectly through an applied force field $\mathbf{f}$:

$$
\begin{align}
\nabla \cdot \boldsymbol{\sigma}' + \mathbf{f} &= 0 \quad (\Omega)\\
\boldsymbol{\sigma}' &= -p' \mathbf{1} + 2\mu \nabla^s \mathbf{v}' \quad (\Omega)\\
\nabla \cdot \mathbf{v}' &= 0 \quad (\Omega)\\
\mathbf{v}', p' \text{ periodic} \quad &(\partial\Omega)
\end{align}
$$

with $v'$, $p'$ periodic on $\partial\Omega$.

*Paper reference: Eq. (4)*

### 3.2 Green Function Solution

The velocity field satisfying the auxiliary problem (4) is formally expressed as:

$$
\mathbf{v}'(\mathbf{x}) = V' + \int_{\Omega} \mathbf{G}(\mathbf{x} - \mathbf{y}) \cdot \mathbf{f}(\mathbf{y}) \, dV_y = V' + (\mathbf{G} * \mathbf{f})(\mathbf{x})
$$

where:
- $\mathbf{G}$ is the Green function of the incompressible Newtonian fluid of viscosity $\mu$ on domain $\Omega$ with periodic boundary conditions
- $V'$ is a constant velocity (the average of $\mathbf{v}'$)
- $*$ denotes spatial convolution

*Paper reference: Eq. (6)*

### 3.3 Admissible Force Fields

The set of admissible force fields that allows retrieval of the solution to the original problem (2) is:

$$
\mathcal{F}(\nabla P) = \left\{ \mathbf{f} \,\middle|\, \mathbf{f}(\mathbf{x}) = -\nabla P \text{ if } \mathbf{x} \in \Omega_f, \,\int_{\Omega} \mathbf{f} dV= \mathbf{0} \right\}
$$

*Paper reference: Eq. (5)*

### 3.4 Variational Principle

From a degenerate minimum stress energy principle, the permeability is obtained from:

$$
\nabla P \cdot K \cdot \nabla P = \inf_{\mathbf{f} \in \mathcal{F}(\nabla P)} \langle \mathbf{f} \cdot (\mathbf{G} * \mathbf{f}) \rangle
$$

*Paper reference: Eq. (7)*

This formulation provides **rigorous upper bounds** on the permeability when the Green operator and support are chosen appropriately.

---

## 4. Discrete Formulation: Trial Force Fields on Voxel Grid

### 4.1 Voxel Grid Discretization

The RVE is discretized on a uniform Cartesian grid of $N = \prod_{i=1}^{d} N_i$ voxels in dimension $d \in \{2, 3\}$.

Voxel position multi-index: $\mathbf{n} = (n_1, \ldots, n_d)$ with $n_i \in [0, \ldots, N_i - 1]$

Indicator function for voxel $\mathbf{n}$: $I_{\mathbf{n}}$ (equals 1 in voxel $\mathbf{n}$, 0 elsewhere)

Three voxel sets are defined:
- $\mathcal{F}$: voxels containing (partially or entirely) fluid phase, cardinality $N_F$
- $\mathcal{S}$: voxels containing only solid phase, cardinality $N_S = N - N_F$
- $\mathcal{B} \subseteq \mathcal{S}$: active support for force unknowns, cardinality $N_B$

*Paper reference: Section 2.3*

### 4.2 Trial Force Field and Decomposition

For any choice of vectors $\{\mathbf{x}_{\mathbf{n}}\}_{\mathbf{n} \in \mathcal{B}}$, the following force field belongs to $\mathcal{F}(\nabla P)$:

$$
\mathbf{f} = \mathbf{f}^0 + \mathbf{f}^x
$$

where:

$$
\mathbf{f}^0 = -\left( \sum_{\mathbf{n} \in \mathcal{F}} I_{\mathbf{n}} + \frac{N_F}{N_B} \sum_{\mathbf{n} \in \mathcal{B}} I_{\mathbf{n}} \right) \nabla P
$$

$$
\mathbf{f}^x = \left( \sum_{\mathbf{n} \in \mathcal{B}} \mathbf{x}_{\mathbf{n}} I_{\mathbf{n}} \right) - \frac{1}{N_B} \left( \sum_{\mathbf{n} \in \mathcal{B}} \mathbf{x}_{\mathbf{n}} \right) \left( \sum_{\mathbf{n} \in \mathcal{B}} I_{\mathbf{n}} \right)
$$

Here $I_{\mathbf{n}}$ is the indicator of voxel $\mathbf{n}$, $\mathbf{f}^x \in \mathcal{F}(0)$, i.e., it has zero average over $\Omega$  
*Paper reference: Eq. (8)*

### 4.3 Discrete Variational Principle

Applying (7) to the decomposition (8) yields an upper bound on permeability for any choice of $\{\mathbf{x}_{\mathbf{n}}\}_{\mathbf{n} \in \mathcal{B}}$:

$$
\nabla P \cdot K \cdot \nabla P \leq \langle \mathbf{f} \cdot (\mathbf{G} * \mathbf{f}) \rangle
$$

*Paper reference: Eq. (9)*

Minimizing with respect to the unknowns yields the discrete optimality condition

$$
\left( G*f \right)_n = \langle G*f \rangle _\mathcal{B} \quad \forall n \in \mathcal{B},
$$

where $(G*f)_n$ denotes the voxel average over voxel $n$ and $\langle\cdot\rangle_\mathcal{B}$ denotes the average over the active support $\mathcal{B}$.

Rewriting with the decomposition $f = f^0 + f^x$:

$$
\left( G*f^x \right)_n - \langle G*f^x \rangle _\mathcal{B} = -\left( G*f^0 \right)_n + \langle G*f^0 \rangle _\mathcal{B} \quad \forall n \in \mathcal{B},
$$

*Paper reference: Eq. (10), Eq. (11)*

### 4.4 Discrete Linear System

Define a compact vector of unknowns $X = \{\mathbf{x}_{\mathbf{n}}\}_{\mathbf{n} \in \mathcal{B}}$. The linear system can be written abstractly as

$$
[\mathbf{A}]\{\mathbf{X}\} = \{\mathbf{b}\}
$$

where the matrix-free operator $\mathbf{A}$ represents

$$
(\mathbf{A}\mathbf{X})_n = (\mathbf{G} * \mathbf{f}^x)_{\mathbf{n}} - \langle \mathbf{G} * \mathbf{f}^x \rangle _{\mathcal{B}} \quad n \in \mathcal{B}
$$

and the right-hand side represents

$$
\mathbf{b}_n = -(\mathbf{G} * \mathbf{f}^0)_{\mathbf{n}} + \langle \mathbf{G} * \mathbf{f}^0 \rangle _{\mathcal{B}} \quad n \in \mathcal{B}
$$

The operator $[\mathbf{A}]$ is **symmetric and semi-definite positive** (all eigenvalues $\geq 0$).  
*Paper reference: Eq. (11), paragraph after*

---

## 5. Matrix-Free Operator Application: Algorithm 1

The matrix $[\mathbf{A}]$ is never explicitly constructed. Instead, the matrix-vector product is computed via Algorithm 1:

**Algorithm 1:** Operation of matrix $[\mathbf{A}]$ on vector $\{\mathbf{X}\}$

```
Input:  {X} = (x_n)_{n ∈ B}
Output: {Y} = [A] · {X}

 1. {X} ← {X} - average({X})                    // Remove mean from input
 2. for all voxel n do
 3.   if n ∈ B then
 4.     F[n] ← X[map(n)]                       // Map support indices to grid
 5.   else
 6.     F[n] ← 0
 7.   end if
 8. end for
 9. F̂ ← DFT(F)                                 // Discrete Fourier Transform
10. for all frequency k do
11.   Û[k] ← Ĝ[k] · F̂[k]                      // Apply Green function in Fourier space
12. end for
13. U ← DFT⁻¹(Û)                               // Inverse DFT
14. for all voxel n ∈ B do
15.   Y[map(n)] ← U[n]                         // Restrict from grid back to support
16. end for
17. {Y} ← {Y} - average({Y})                    // Remove mean from output
```

*Paper reference: Algorithm 1, Section 2.4*

**Key implementation points:**
- FFT is used to evaluate the Green convolution efficiently (steps 9-13)
- Support-to-grid mapping (step 4) and reverse mapping (step 15) handle the restriction to active support $\mathcal{B}$
- Mean removal (steps 1, 17) maintains consistency with the zero-mean constraint on $\mathbf{f}^x$
- Computational complexity: $O(N \log N)$ per iteration due to FFT

---

## 6. Discretized Green Operators

The paper investigates seven discretizations of the Green function. Only the **energy-consistent discretization** ensures the rigorous upper-bound property when the support choice is FS or FI (defined in Section 7).

### 6.1 Continuous Green Function in Fourier Space

For an incompressible Newtonian fluid of viscosity $\mu$ in an infinite domain:

$$
\hat{\mathbf{G}}(\mathbf{q}) = \begin{cases}
\frac{1}{\mu \|\mathbf{q}\|^2} \left( \mathbf{1} - \frac{\mathbf{q} \otimes \mathbf{q}}{\|\mathbf{q}\|^2} \right) & \text{if } \mathbf{q} \neq 0\\
0 & \text{if } \mathbf{q} = 0
\end{cases}
$$

*Paper reference: Eq. (12)*

where $\mathbf{q}$ is the frequency vector in Fourier space.

For $L$-periodicity with $L_i$ period in direction $i$, the allowed frequencies are:

$$
\mathbf{q}_{\mathbf{k}} = \sum_{i=1}^{d} \frac{2\pi k_i}{L_i} \mathbf{e}_i
$$

*Paper reference: Eq. (13)*

### 6.2 Energy-Consistent Discretization

The energy-consistent discretized Green function is defined as:

$$
\hat{\mathbf{G}}^{N,E}_{\mathbf{k}} = \sum_{\mathbf{p} \in \mathbb{Z}^d} \left( \prod_{i=1}^{d} \text{sinc}^2\left(\pi \frac{k_i + p_i N_i}{N_i}\right) \right) \hat{\mathbf{G}}\left(\mathbf{q}_{k_1 + p_1 N_1, \ldots, k_d + p_d N_d}\right)
$$

with $sinc(x) = sin(x)/x$, and with $\hat{G}(0) = 0$

*Paper reference: Eq. (14), Section 2.5*

**Properties:**
- This discretization is the **only one that guarantees the upper-bound property** of permeability for support choices FS or FI
- Guarantees energy consistency: the discrete problem preserves the variational structure
- Must be pre-computed before iteration (except for other discretizations, which can be computed on-the-fly)
- Requires numerical summation of the infinite series (see paper Appendix A for details)

### 6.3 Alternative Discretizations (Not Recommended for Rigor)

For reference and comparison, the paper also investigates six alternative discretizations:

1. **Truncated** (Eq. 15): Direct evaluation of continuous operator at coarse frequencies; introduces discontinuities and Gibbs oscillations
2. **Filtered** (Eq. 16): Compromise between truncated and energy-consistent; better convergence than truncated
3. **Centered finite difference** (Eq. 18): Symmetric second-order finite difference; checkerboarding artifacts
4. **Forward-backward finite difference** (Eq. 20): Requires staggered grid; smooth fields but complex implementation
5. **Rotated-basis finite difference** (Eq. 21): More local accuracy; introduces checkerboarding; poor convergence in 3D
6. **Hybrid-stride centered** (Eq. 23): Locally accurate Laplacian; good convergence; suitable compromise for practical use

*Paper reference: Table 1, Equations (15)-(23), Section 2.5*

**Recommendation for implementation:** Use the energy-consistent discretization (Eq. 14) to maintain mathematical rigor and guarantee upper-bound status.

---

## 7. Support Choice: Active Voxel Set $\mathcal{B}$

The paper presents four choices for the active support subset $\mathcal{B} \subseteq \mathcal{S}$:

### 7.1 Rigorously Upper-Bounding Choices (Recommended)

**FS (Force throughout Solid):**
- $\mathcal{B} = \mathcal{S}$ (all solid voxels)
- Dimensions: $d \times N_S$ unknowns
- Property: Rigorous upper bound (with energy-consistent Green operator)
- Drawback: Large problem; poor condition number; many iterations required

**FI (Force at Interface, Recommended):**
- $\mathcal{B}$ = solid voxels having at least one neighbor (sharing vertex, edge, or face) not in $\mathcal{S}$
- Dimensions: $d \times N_B$ unknowns, where $N_B \ll N_S$ for fine grids
- Property: Rigorous upper bound (with energy-consistent Green operator)
- Advantage: Much smaller problem; better condition number; faster convergence
- **Recommendation:** Use FI for efficiency and mathematical rigor

*Paper reference: Section 2.3, Fig. 1, discussion in Section 3.2.1*

### 7.2 Alternative Choices (May Lose Upper-Bound Guarantee)

**CS (Center-based throughout Solid):**
- Voxels whose center lies in solid phase (not all voxels crossing the interface)
- Does not strictly fulfill requirements (5); may underestimate permeability
- Useful for correcting phase discretization bias

**CI (Center-based at Interface):**
- Voxels whose center lies in solid phase AND that neighbor fluid-centered voxels
- Does not strictly fulfill requirements (5); may not give rigorous upper bound
- Useful for correcting phase discretization bias

*Paper reference: Section 2.3, Fig. 1*

**Recommended choice for first implementation: FI (Force at Interface) with energy-consistent Green operator**

---

## 8. Variable Placement Conventions

The discretization of the Green operator and the force placement on the voxel grid must be coordinated. Three placement conventions are investigated:

### 8.1 @center
Forces and velocities placed at voxel centers; suitable for energy-consistent, truncated, filtered, and rotated operators.

### 8.2 @vertex
Forces and velocities placed at voxel vertices; suitable for rotated and hybrid operators to exactly enforce no-slip conditions at geometry corners.

### 8.3 @face
Forces and velocities placed at voxel face centers (staggered grid); required for forward-backward operator to enforce mass conservation exactly at voxel centers.

*Paper reference: Section 2.6, Fig. 3*

**Recommendation for first implementation:** Use @center placement with energy-consistent Green operator (default in Section 6.2).

---

## 9. Velocity Recovery

After solving the linear system (11) for the unknown force field $\{\mathbf{x}_{\mathbf{n}}\}_{\mathbf{n} \in \mathcal{B}}$, the velocity field is recovered using the Green function:

### 9.1 Solved Force Field

Combining the solved $\mathbf{f}^x$ with the known $\mathbf{f}^0$:

$$
\mathbf{f} = \mathbf{f}^0 + \mathbf{f}^x
$$

### 9.2 Velocity Field Recovery

The velocity field at each voxel is obtained through Green convolution:

$$
\mathbf{v}'(x) = V' + \left( \mathbf{G} * \mathbf{f} \right)(x)
$$

To fix $V'$, Bignonnet chooses it such that the average velocity over the optimization support $\mathcal{B}$ is zero:

$$
V' = - \left( G*f \right) _\mathcal{B},
$$

where $\left( G*f \right)_{\mathcal{B}}$ is the average of $G*f$ over voxels in $\mathcal{B}$.

**Implementation via Algorithm 1:**
- Construct the full force field $\mathbf{f}$ on the grid (map $\mathbf{f}^x$ from support to grid, add $\mathbf{f}^0$)
- Compute $\hat{\mathbf{F}} = \text{DFT}(\mathbf{f})$
- For each frequency: $\hat{\mathbf{U}}[\mathbf{k}] = \hat{\mathbf{G}}^{N,E}_{\mathbf{k}} \cdot \hat{\mathbf{F}}[\mathbf{k}]$
- Compute $\mathbf{U} = \text{DFT}^{-1}(\hat{\mathbf{U}})$
- Compute its average over $\mathcal{B}$.
- Set $V'=-\langle U \rangle _\mathcal{B}$.
- Define the final velocity as $v_n = U_n + V'$

*Paper reference: Eq. (6), Eq. (10), Sections 2.4-2.6*

---

## 10. Permeability Tensor Extraction

The homogenized permeability tensor is extracted by solving the system for canonical macroscopic loading cases:

### 10.1 Three Loading Cases

For a Cartesian coordinate system, apply three independent unit pressure gradients in directions $i = 1, 2, 3$:

$$
\nabla P = \mathbf{e}_i \quad (i = 1, 2, 3)
$$

For each case $i$:
1. Set up and solve the linear system (11) with MINRES
2. Recover the velocity field $\mathbf{v}^{(i)}$ using Algorithm 1 and Section 9
3. Compute the macroscopic filtration velocity as the volume average:

$$
V_i = \langle v_i \rangle = \frac{1}{|\Omega|} \int_{\Omega_f} v_i \, dV
$$

### 10.2 Permeability Components

From Darcy's law $V = -K \cdot \nabla P$, with $\nabla P = \mathbf{e}_i$:

$$
K_{i,j} = -V_j^{(i)}
$$

This yields the full permeability tensor $\mathbf{K}$.

*Paper reference: Section 2.1, Eq. (1), discussion in Section 4*

### 10.3 Symmetry Check

Due to symmetry of the variational formulation and periodic boundary conditions, the tensor $\mathbf{K}$ is symmetric: $K_{i,j} = K_{j,i}$.

---

## 11. Linear Solver: MINRES

The discrete linear system (11) is solved using the MINRES (Minimum Residual) iterative method.

### 11.1 Solver Properties

**Why MINRES:**
- The operator $[\mathbf{A}]$ is symmetric and semi-definite positive (all eigenvalues $\geq 0$)
- Conjugate Gradient (CG) is not suitable because the operator is semi-definite, not positive definite (zero eigenvalue due to overall mean constraint)
- MINRES is the appropriate choice for symmetric semi-definite systems

*Paper reference: Section 2.4, sentence: "Both the Conjugate Gradient and the Minres solvers [34] may apply. After a series of tests, Minres appeared much better behaved than Conjugate Gradient – as expected for this type of linear system [35] – and is adopted."*

### 11.2 Convergence Criterion

The stopping criterion is based on the residual norm:

$$
\|\{\mathbf{r}\}\|_2 \leq \eta \|\{\mathbf{b}\}\|_2
$$

where $\{\mathbf{r}\} = [\mathbf{A}]\{\mathbf{x}\} - \{\mathbf{b}\}$ is the residual vector.

**Target tolerance:** $\eta = 10^{-10}$ (relative residual tolerance)

*Paper reference: Section 2.4, text: "The convergence criterion used on the norm of the residual $\{\mathbf{r}\} = [\mathbf{A}]\{\mathbf{x}\} - \{\mathbf{b}\}$ is $\|\{\mathbf{r}\}\|_2 \leq \eta \|\{\mathbf{b}\}\|_2$ where $\eta = 10^{-10}$ in all presented simulations."*

### 11.3 Solver Implementation

- Use a standard MINRES implementation (available in libraries like PETSc, Eigen, or FFTW-compatible solvers)
- In Algorithm 1, compute matrix-vector products on demand
- Pre-compute the energy-consistent Green operator (Eq. 14) before starting iterations
- No explicit matrix construction needed

---

## 12. Numerical Algorithm Overview

**High-level solution process:**

1. **Setup phase:**
   - Discretize pore geometry on uniform grid
   - Identify fluid voxels ($\mathcal{F}$) and solid voxels ($\mathcal{S}$)
   - Define active support $\mathcal{B}$ (use FI for recommended choice)
   - Pre-compute energy-consistent Green operator $\hat{\mathbf{G}}^{N,E}_{\mathbf{k}}$ via Eq. (14)

2. **For each canonical loading case** ($\nabla P = \mathbf{e}_i$, $i = 1, 2, 3$):
   - Construct $\mathbf{f}^0$ from Eq. (8)
   - Set up RHS $\{\mathbf{b}\}$ via Algorithm 1 applied to $\mathbf{f}^0$
   - Solve $[\mathbf{A}]\{\mathbf{x}\} = \{\mathbf{b}\}$ using MINRES with $\eta = 10^{-10}$
   - Recover velocity field via Green convolution (Eq. 6, Section 9)
   - Compute permeability column $\mathbf{K}_{:,i} = -\langle \mathbf{v}^{(i)} \rangle$

3. **Output:**
   - Homogenized permeability tensor $\mathbf{K}$ (3×3 symmetric matrix)
   - Optional: local velocity fields at each voxel
   - Optional: RVE size statistics (see Section 4 of paper)

*Paper reference: Sections 2-3, Algorithm 1*

---

## 13. Mathematical Assumptions and Constraints

### 13.1 Fundamental Assumptions

1. **Periodic domain:** RVE has periodic boundary conditions in all directions
2. **Stokes flow:** Incompressible Newtonian fluid at low Reynolds number
3. **No-slip condition:** Velocity is zero on solid-fluid interface $\Gamma$
4. **Uniform viscosity:** Fluid has constant viscosity $\mu$ throughout pore space
5. **Linear constitutive law:** Stress-strain relation $\boldsymbol{\sigma} = -p\mathbf{1} + 2 \mu \nabla^s \mathbf{v}$
6. **Small-scale separation:** RVE size $L$ and voxel size $H = L/N_i$ are such that $H \ll L$ for accuracy

*Paper reference: Section 2.1, Eq. (2)*

### 13.2 Discretization Choices

1. **Uniform Cartesian grid:** Voxel-based discretization with equal spacing in each direction (may be non-isotropic: $H_i \neq H_j$)
2. **Energy-consistent Green operator:** Ensures mathematical upper-bound status (Eq. 14)
3. **FI support choice:** Forces only at interface voxels (balance between accuracy and efficiency)
4. **@center placement:** Forces and velocities at voxel centers (simplest and most standard)
5. **MINRES solver:** Appropriate for symmetric semi-definite operator (Section 11)
6. **Relative residual tolerance $\eta = 10^{-10}$:** Target for linear system accuracy

### 13.3 Implementation Constraints

1. All voxel-wise functions (force, velocity, indicator) are piecewise constant within each voxel
2. FFT operations require the domain to be periodic; non-periodic boundary conditions require special treatment (not implemented in this paper)
3. Voxel labeling (solid/fluid) is binary; no multi-phase or fractional-volume treatment
4. Memory scales as $O(d N + d N_B)$ where $N$ is total voxels and $N_B \ll N$ is active support
5. Computational cost per MINRES iteration is $O(N \log N)$ due to FFT

*Paper reference: Sections 2-3, Algorithm 1, Section 3.4*

---

## 14. Explicit Non-Goals and Out-of-Scope Items

The following are **NOT** implemented in this Stage 1 plan:

1. **Ad hoc masked velocity operator:** The operator is derived from the variational framework (Eq. 7-11), not from a heuristic masked Stokes fixed-point map
2. **Simplified inverse-Laplacian Stokes prototype:** The Green function must be discretized consistently with the variational formulation (Eq. 14)
3. **GMRES-first formulation:** The operator is symmetric, so MINRES (not GMRES) is the correct choice
4. **Post-hoc $h^2$ scaling trick:** Any dimensional corrections must arise naturally from the discrete-to-continuum analysis, not from external parameter tuning
5. **Alternative force placement conventions:** The first implementation uses @center; @vertex and @face are investigated but not required for the baseline
6. **RVE size estimation:** While the paper includes RVE methodology (Section 4), that is a secondary feature for this initial Stage 1 plan
7. **Non-periodic domain boundaries:** The formulation requires periodic boundary conditions; handling open or Dirichlet/Neumann boundaries is out of scope
8. **Non-uniform grids:** All directions use uniform spacing (though $H_i$ and $L_i$ may differ by coordinate direction)

*Paper reference: Throughout, particularly Section 2 on constraints and Section 3 on performance comparisons*

---

## 15. Key Mathematical Properties to Verify During Implementation

1. **Operator symmetry:** Verify numerically that $[\mathbf{A}]$ is symmetric (within machine precision)
2. **Positive semi-definiteness:** Check that eigenvalue spectrum is $\geq 0$ (or near zero for numerical precision)
3. **Permeability upper bound:** Computed $K$ should be $\geq$ reference permeability (for known test cases)
4. **Permeability symmetry:** The tensor $\mathbf{K}$ should satisfy $K_{i,j} = K_{j,i}$
5. **Convergence rate:** Energy-consistent operator should achieve $O(H/D)$ convergence in permeability error (Section 3)
6. **MINRES iterations:** Number of iterations should scale well with refinement (fewer for FI than FS support)
7. **Velocity accuracy:** Local velocity fields should show no significant oscillations (checkerboarding-free with energy-consistent operator)

*Paper reference: Sections 3.2-3.3*

---

## 16. References and Paper Structure

**Full Citation:**  
François Bignonnet. "Efficient FFT-based upscaling of the permeability of porous media discretized on uniform grids with estimation of RVE size." *Computer Methods in Applied Mechanics and Engineering*, Volume 369 (2020) 113237. DOI: 10.1016/j.cma.2020.113237

**Key sections in paper:**
- **Section 2.1:** Periodic homogenization of permeability from Stokes (Eqs. 1-3)
- **Section 2.2:** Variational framework based on trial force fields (Eqs. 4-7)
- **Section 2.3:** Discrete trial force fields and support choices (Eqs. 8-10)
- **Section 2.4:** Numerical method, Algorithm 1, MINRES solver (Eq. 11, Algorithm 1)
- **Section 2.5:** Seven discretizations of Green operator (Eqs. 12-23, Table 1)
- **Section 2.6:** Variable placement conventions (@center, @vertex, @face)
- **Section 3:** Numerical validation on test microstructures (2D square, 2D circle, 3D Voronoi)
- **Section 4:** RVE estimation methodology (not required for Stage 1)

---

## Summary

This Stage 1 plan provides the exact mathematical formulation from Bignonnet (2020) for implementing an FFT-based permeability solver using:
- **Unknown:** Discrete trial force field on interface voxels
- **Variational framework:** Minimization of stress energy to bound permeability
- **Discretization:** Voxel-wise constant force fields with energy-consistent Green operator
- **Operator:** Matrix-free via FFT-based Green convolution (Algorithm 1)
- **Solver:** MINRES with relative residual tolerance $10^{-10}$
- **Output:** Homogenized permeability tensor via canonical load cases

All formulations are equation-faithful references to the published paper. No invented formulas.

**Status:** Ready for Stage 2 (detailed design and implementation in C++20).
