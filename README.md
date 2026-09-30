# VQBR: Variational Quantum Bayesian Regression

**Via Measurement-Based MAP Direction Recovery**

Official research implementation by **Van Tien Nguyen** and **Panagiotis Markopoulos**, accepted for publication at the **2026 IEEE International Conference on Quantum Computing and Engineering (QCE)**.

[Presentation slides](presentation/VQBR_QCE2026_Slides.pdf) · [Experiment notes](doc/REPRODUCIBILITY.md)

VQBR is a hybrid quantum-classical method for maximum a posteriori (MAP) estimation in Bayesian linear regression. It encodes a candidate weight direction in a variational quantum state, eliminates its scalar magnitude analytically, and estimates the objective through row-wise overlaps and Pauli measurements without explicitly forming the Gram matrix.

- The direction uses `ceil(log2(D))` qubits; overlap circuits add one ancilla.
- Scale elimination is exact within the chosen ansatz family. Recovery of the unrestricted MAP solution depends on circuit expressivity and optimization.
- The experiments include statevector simulation, finite-shot sampling, ansatz-depth studies, and one IBM hardware demonstration with zero-noise extrapolation (ZNE).
- This work estimates a MAP point, not posterior predictive uncertainty, and does not claim a runtime advantage over matrix-free classical solvers.

## Installation

Use **Python 3.11**, the version used to validate the dependency set below. Run all commands from the repository root in a Bash-compatible shell.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start

Run a small synthetic experiment locally, without downloading data or configuring an IBM account:

```bash
python script/exp/synthetic_experiemnt.py \
  --N 50 --D 8 --num-seeds 1 \
  --vqbr-maxiter 30 --vqbr-no-shot-noise \
  --no-log-training --run-dir runs/quickstart
```

The filename `synthetic_experiemnt.py` retains its original spelling. This short run checks the workflow; its small optimization budget is not intended to reproduce the paper's accuracy.

Outputs appear in `runs/quickstart/`:

| File | Contents |
| --- | --- |
| `summary_metrics.csv` | Aggregate VQBR and direct MAP metrics |
| `per_seed_metrics.csv` | Cosine similarity, RMSE, and R² for each seed/method |
| `results.json` | Detailed results and optimization histories |
| `training_log.csv` | Per-evaluation optimization records |
| `run_config.json` | Arguments and invocation used for the run |
| `seed_artifacts/` | Saved arrays for the splits, prior, and fitted models |
| `run.log` | Console output |

Plot the convergence history:

```bash
python script/plot/synthetic_plot.py \
  --results-path runs/quickstart/results.json \
  --training-csv-path runs/quickstart/training_log.csv \
  --objective-plot-path runs/quickstart/objective.png --dpi 150
```

`runs/` is ignored by Git. Choose a new `--run-dir` to preserve a previous run; reusing a directory can overwrite its files. Omitting this flag writes a timestamped directory under `results/` by default.

## Experiments

### Synthetic benchmarks

These commands match the principal archived runs' iteration budgets. They generate data internally and use seeds 0–19, the 0.2/0.6/0.2 prior/train/test split, and a two-repetition ansatz.

```bash
python script/exp/synthetic_experiemnt.py \
  --N 200 --D 8 --num-seeds 20 --vqbr-maxiter 300 \
  --vqbr-no-shot-noise --run-dir runs/synthetic-d8

python script/exp/synthetic_experiemnt.py \
  --N 200 --D 16 --num-seeds 20 --vqbr-maxiter 1000 \
  --vqbr-no-shot-noise --run-dir runs/synthetic-d16

python script/exp/synthetic_experiemnt.py \
  --N 200 --D 32 --num-seeds 20 --vqbr-maxiter 2000 \
  --vqbr-no-shot-noise --run-dir runs/synthetic-d32
```

**Reproducibility note:** the archived configurations and script defaults differ from the manuscript's general setup paragraph, including bootstrap size, ridge regularization, COBYLA tolerance, gate list, and some iteration budgets. See [the configuration comparison](doc/REPRODUCIBILITY.md) before attempting an exact reproduction. The commands above rerun the archived configuration choices; identical numerical results are not guaranteed across dependency versions.

### Energy Efficiency

Download the **Energy Efficiency** dataset by A. Tsanas and A. Xifara from its [UCI dataset page](https://archive.ics.uci.edu/dataset/242/energy+efficiency). Extract `ENB2012_data.xlsx` to `data/real/energy/` (create the directory if needed). The dataset contains 768 examples, eight features, and heating/cooling targets; this experiment predicts heating load (`Y1`). Raw real-world datasets are ignored by Git and must be obtained separately.

```bash
python script/exp/energy_experiment.py \
  --dataset-path data/real/energy/ENB2012_data.xlsx \
  --target-col Y1 --num-seeds 20 --vqbr-maxiter 300 \
  --vqbr-no-shot-noise --run-dir runs/energy

python script/plot/energy_plot.py \
  --results-path runs/energy/results.json \
  --training-csv-path runs/energy/training_log.csv \
  --objective-plot-path runs/energy/objective.png --dpi 150
```

### Ansatz depth and finite-shot sampling

```bash
python script/exp/ablation_ansatz_expressibility.py \
  --N 200 --D 16 --num-seeds 20 --reps-min 1 --reps-max 5 \
  --vqbr-maxiter 1000 --vqbr-no-shot-noise --run-dir runs/ansatz-depth

python script/exp/ablation_shot_noise.py \
  --N 200 --D 8 --num-seeds 20 --vqbr-maxiter 300 \
  --include-analytic --shots-list 256,1024,4096,16384 \
  --run-dir runs/shot-noise
```

Finite-shot and full multi-seed runs can take substantially longer than the quick start. Each script exposes its options with `--help`. Matching plotting scripts are in `script/plot/`; pass explicit input/output paths for the run you want to inspect.

### Hardware and additional experiments

For a local noisy simulation with the hardware experiment driver:

```bash
python script/exp/ablation_real_hardware.py \
  --hardware-runner aer_noise --N 50 --D 8 --seeds 0 \
  --vqbr-maxiter 30 --ibm-shots 1024 --run-dir runs/aer-noise
```

For IBM hardware, create the local, Git-ignored `config/ibm_config.json` with your `token`, `channel`, and `instance` values. Use the backend available to your account:

```bash
python script/exp/ablation_real_hardware.py \
  --hardware-runner ibm --ibm-config config/ibm_config.json \
  --ibm-backend YOUR_BACKEND --N 50 --D 8 --seeds 0 \
  --vqbr-maxiter 100 --ibm-shots 1024 --run-dir runs/ibm-hardware
```

The second command submits remote jobs and requires IBM Quantum access. Hardware availability, calibration, queue time, and account usage affect a rerun. The archived hardware demonstration used `ibm_kingston`.

The repository also contains Naval and YearPredictionMSD experiment drivers, a conjugate-gradient baseline, a hardware evaluation probe, and a debugging notebook. These are additional research artifacts; they are not all part of the accepted manuscript's validation. Consult their `--help` output for dataset paths and options.

## Results reported in the manuscript

Simulation values below are mean ± standard deviation over 20 seeds. The hardware value is from one proof-of-concept run.

| Setting | Cosine similarity to direct MAP |
| --- | ---: |
| Synthetic, D = 8, p = 2 | 0.998 ± 0.005 |
| Energy Efficiency, D = 8, p = 2 | 0.927 ± 0.039 |
| Synthetic, D = 16, p = 5 | 0.9999 ± 0.0001 |
| IBM hardware with ZNE, D = 8 | 0.9478 |

The accepted manuscript provides predictive metrics and limitations. Archived configurations, metrics, and figures are in [`results/`](results/); these quoted values have not been recomputed by the quick start.

## Repository layout

```text
src/
  vqbr/                 Variational quantum regression implementation and demos
  closed_form/          Direct MAP reference and unit tests
  conjugate_gd/         Conjugate-gradient reference and unit tests
script/
  exp/                  Experiment command-line entry points
  plot/                 Plotting and table-generation scripts
data/
  synthetic/            Synthetic-data generator and archived generated datasets
  real/                 Local downloads (raw datasets are ignored)
results/                Archived experiment outputs and run configurations
notebooks/              Step-by-step debugging notebook
presentation/           QCE presentation slides
doc/                    Reproducibility notes and slide figure sources
requirements.txt        Experiment, plotting, test, and hardware dependencies
```

## Validation

```bash
python -m pytest -q
```

The existing unit tests cover the direct MAP and conjugate-gradient implementations. Use the quick-start experiment and plotting command above to check the VQBR workflow as well. Files named `vqbr_test.py` and `vqbr_synthetic_test.py` are standalone demonstration scripts, not automated pytest coverage of the quantum algorithm.

## Citation

```bibtex
@inproceedings{nguyen2026vqbr,
  author    = {Nguyen, Van Tien and Markopoulos, Panagiotis},
  title     = {{VQBR}: Variational Quantum Bayesian Regression via Measurement-Based MAP Direction Recovery},
  booktitle = {2026 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  year      = {2026},
  note      = {Accepted for publication}
}
```

## Contact

**Van Tien Nguyen**

[tien.nguyen@utsa.edu](mailto:tien.nguyen@utsa.edu) · [vantn.prof@gmail.com](mailto:vantn.prof@gmail.com)

[Personal website](https://vantnprof.github.io)
