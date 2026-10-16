# TAMU_GRASP_Kernel_Generator

GRASP-format kernels (`kernels_299_00/11/12/22/33/34/44.txt`, `grid1.dat`) for the TAMUdust2020
irregular-hexahedra ensembles (Saito et al. 2021, JAS 78, 2089), built from `tamudust2020create`
output. Author: Greema Regmi. Correction of October 2026: Reed Espinosa.

| File | Role |
|---|---|
| `GRASPhexKernel.py` | `RunTAMUdust2020` (runs TAMUdust2020 and builds the kernels), `volume_kernels`, `ensemble_sphericity`, `Phase_Function_299` (writes the files) |
| `generateGRASPhexKernel.py` | Driver: size and refractive-index grids, loop over sphericities |
| `test_GRASPhexKernel.py` | Checks of the kernel formulas and element files (no TAMUdust2020 run needed) |
| `Kernels_hexahedral.ipynb` | Original notebook (formula corrected; `GRASPhexKernel.py` is the maintained version) |
| `Kernels_spherical.ipynb` | **Superseded**; see below |

## What the kernels contain

At the reference wavelength 0.34 µm, on a log-uniform grid of volume-equivalent radii with step dlnr:

- EXTINCTION, ABSORPTION: C/V × dlnr [1/µm] = Qe·A/V × dlnr, where Qe = C_ext/A is TAMUdust2020's extinction efficiency and A its orientation-averaged projected area.
- Element ij: F_ij × C_sca/V × dlnr, with ½∫F11 sinθ dθ = 1. TAMUdust2020 gives P_ij/P11; element 34 for GRASP is −P43 × P11 × C_sca/V × dlnr, written as `kernels_299_34.txt`.
- `kernel_info.json` (written with `kernels_299_00.txt`): nominal and ensemble sphericity, α = A/(π·r_v²), dlnr, size descriptor.

## October 2026 corrections

1. **Kernel normalization.** Earlier versions computed the extinction and absorption kernels as `Qe*3*dlnr/(4*r_v)` = Qe·π·r_v²/V·dlnr, i.e. TAMUdust2020's true-area Qe with the cross section of the volume-equivalent sphere. All kernels made that way (extinction, absorption and all scattering-matrix elements, which are scaled by b_ext − b_abs) are low by the sphericity Ψ = π·r_v²/A. Normalized quantities (SSA, P_ij/P11) are unaffected. All size descriptors now use C/V = Qe·A/V (`volume_kernels`), and a build-time check asserts that C_ext/(π·r_v²) approaches 2A/(π·r_v²) at the largest size.
   - `TAMUratioA2V_ER` divided by 4V (a factor 4 too small); fixed.
   - `ER` was already right, since 3/(4·r_eff) = A/V.
2. **Element 34.** `generateGRASPhexKernel.py` wrote +P43 to `kernels_299_43.txt`, which GRASP does not read. It now writes −P43 to `kernels_299_34.txt`, the convention of the 2023/2024 kernel sets.
3. **Smaller fixes.**
   - `size_descriptor=None` ran two branches; it now means `'VER'`.
   - The results pickle stored the built-in type `dict` instead of the results.
   - The driver called `Phase_Function_299(Dict1, 00, ...)`, which wrote nothing; it now writes `grid1.dat` (`'grid'`).
4. **Sphere kernels.** `Kernels_spherical.ipynb` produced `kernels_100_*` that are 6.04% too high (a `dlnr` taken from the first step of the TAMUdust2020 v1.0 radii) and invalid above x ≈ 583. Sphere kernels on this or any other GRASP grid are now computed directly with Lorenz–Mie code by `GEOSmie_TOA/graspkernels` (config `input_files/kernelgeneration/grasp_sphere_tamu273.json` for the 273-node TAMUdust2020 grid).

Corrected kernel sets: `NIP_Morphology_Polarimetry/TAMUdust2020_kernels/TAMUdust2020-in-GRASP-kernel-format_V5-2026-10` (S = 0.7, with the direct-Mie sphere kernel). These were made by multiplying the 2024 kernels by α with `GEOSmie_TOA/runkernelscaling.py`. Details: `STG_AerosolModelExchange/Mian/hexahedra_kernel_findings.md`.

## Correcting kernels made with the old formula

Multiply every value in `kernels_299_00/11/12/22/33/34/44.txt` by α = A/(π·r_v²) of the ensemble (e.g. with `GEOSmie_TOA/runkernelscaling.py`). TAMUdust2020 (v1.0 and `master`) interpolates A, V and the cross sections linearly in ln S between its sphericity bins (0.695, 0.712, 0.7378, 0.7552, 0.7729, 0.785). The ensemble it delivers therefore has a sphericity slightly different from the nominal value between bins. α from the database A and V:

| nominal S | 0.695 | 0.704 | 0.713 | 0.722 | 0.731 | 0.740 | 0.749 | 0.758 | 0.767 | 0.776 | 0.785 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| α | 1.43885 | 1.41895 | 1.40198 | 1.38189 | 1.36563 | 1.35059 | 1.33368 | 1.31814 | 1.30211 | 1.28789 | 1.27388 |

At S = 0.7, α = 1.42726 (Ψ = 0.70064). Regenerating with the corrected code gives the same result directly.

TAMUdust2020 v1.0 Akima-interpolates in size, so its radii are not exactly log-uniform; use `master`. `volume_kernels` asserts a log-uniform grid.
