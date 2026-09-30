# Reproducibility notes

The accepted manuscript, current scripts, and archived results are related research artifacts, but their configuration descriptions are not fully aligned. Preserve the saved results as historical evidence; do not silently relabel them as runs with the manuscript's general settings.

## Configuration differences

| Parameter | Manuscript §4.1 | Current defaults / principal archived runs |
| --- | --- | --- |
| Prior bootstrap samples | 200 | 64 |
| Prior ridge parameter | 1e-3 | 1e-6 |
| COBYLA tolerance | 2e-3 | 1e-8 |
| Maximum evaluations | 1000 in the general setup | Synthetic D=8: 300; D=16: 1000; D=32: 2000; Energy: 300 |
| EfficientSU2 gates | Described as Ry rotations | `ry,x` (parameterized Ry and fixed X gates) |

The synthetic, depth, and shot-noise drivers enforce `ry,x`; the Energy driver exposes `--vqbr-su2-gates`. A run matching the manuscript's prose therefore requires resolving the gate-description difference as well as setting numerical flags. The README's benchmark commands intentionally retain the archived configuration choices.

For example, the numeric settings in the manuscript can be requested with:

```text
--prior-bootstrap-samples 200 --prior-ridge 1e-3
--vqbr-cobyla-tol 2e-3 --vqbr-maxiter 1000
```

These flags alone do not establish an exact reproduction of the paper. The manuscript/configuration differences need author reconciliation before claiming exact reproducibility.

## Principal archives

Paths below are relative to `results/`. Each experiment directory contains its own `run_config.json` and metrics.

| Experiment | Directory |
| --- | --- |
| Synthetic D=8 | `synthetic/synthetic_N200_D8_seeds20_maxiter300_cobyla_log_ratio_analytic_reps2_gatesry-x_entlinear_20260318_085932` |
| Synthetic D=16 | `synthetic/synthetic_N200_D16_seeds20_maxiter1000_cobyla_log_ratio_analytic_reps2_gatesry-x_entlinear_20260318_222602` |
| Synthetic D=32 | `synthetic/synthetic_N200_D32_seeds20_maxiter2000_cobyla_log_ratio_analytic_reps2_gatesry-x_entlinear_20260318_222505` |
| Energy | `energy/energy_enb2012_data_y1_maxsall_seeds20_maxiter300_cobyla_log_ratio_analytic_reps2_gatesry-x_entlinear_20260318_154127` |
| Ansatz depth | `synthetic/ablation_ansatz_expressibility_N200_D16_seeds20_maxiter1000_cobyla_log_ratio_analytic_reps1-5_gatesry-x_entlinear_20260325_120354` |
| Shot noise | `synthetic/ablation_shot_noise_N200_D8_seeds20_maxiter300_cobyla_log_ratio_analytic-shots256-shots1024-shots4096-shots16384_gatesry-x_entlinear_20260325_124041` |
| IBM hardware | `synthetic/ablation_real_hardware_ibm_kingston` |

Some JSON artifacts contain absolute paths from the original machine. For plotting, pass the local `results.json` and CSV paths explicitly, as shown in the README. Historical `command` entries may omit the Python executable; invoke the corresponding script with `python`. Optional CG replay paths may need relocation when using archives copied from another machine.

## Data and validation scope

Synthetic experiments generate their own data, so the quick start needs no download. The tracked `data/synthetic/generated/` collection is a separate archive produced by `data/synthetic/data_synthetic.py`.

Real-world raw data is intentionally excluded from Git. Obtain Energy Efficiency from the [UCI source](https://archive.ics.uci.edu/dataset/242/energy+efficiency), and place the workbook at the path documented in the README. Local Naval and YearPredictionMSD downloads are likewise not included in a fresh clone.

`requirements.txt` records a tested Python 3.11 dependency set for running this repository. It does not reconstruct the historical software environment. Unit tests validate classical references; a small VQBR experiment checks execution and artifact generation. Neither substitutes for rerunning the full 20-seed benchmarks or repeating remote hardware experiments.
