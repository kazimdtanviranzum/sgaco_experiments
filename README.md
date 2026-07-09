# SG-ACO Experiments

This repository contains the code, benchmark metadata, run outputs, statistical tests, and figures for the experimental study of **SHAP-Guided Ant Colony Optimization (SG-ACO)** for the Job-Shop Scheduling Problem (JSSP).

SG-ACO uses a gradient-boosted surrogate model and signed TreeSHAP feature attributions to guide the construction heuristic inside Ant Colony Optimization. The experiments compare SG-ACO against standard ACO variants, tabu search, and deterministic dispatching rules on named public JSSP benchmark instances.

## Repository contents

| File | Description |
|---|---|
| `engine.py` | JSSP decoder, feature extraction, schedule construction, dispatching rules, ACO kernels, and tabu search. |
| `methods.py` | SG-ACO method logic: corpus generation, surrogate training, TreeSHAP signed weights, and ACO variants. |
| `run_resumable.py` | Resumable experiment runner for the full stochastic grid. |
| `analyze.py` | Generates summary tables, statistical tests, ablation results, timing tables, and figures. |
| `make_report.py` | Produces a manuscript-ready experimental report from the generated outputs. |
| `instances.json` | Benchmark metadata, sizes, best-known/optimal values, and expected instance paths. |
| `inst_info.json` | Per-instance surrogate information, corpus best values, training sizes, timings, and SHAP weights. |
| `results.csv` | Per-run results for each instance, method, and seed. |
| `summary.csv` | Per-instance summary statistics: mean, standard deviation, minimum, and BKS gap. |
| `wilcoxon.csv` | Per-instance Wilcoxon signed-rank tests with Holm correction. |
| `wilcoxon_pooled.csv` | Pooled Wilcoxon tests over BKS-normalized paired blocks. |
| `friedman.json` | Friedman test statistic, p-value, number of blocks, and average ranks. |
| `nemenyi.csv` | Nemenyi post-hoc p-values. |
| `ablation.csv` | Signed-SHAP vs magnitude-only SHAP ablation results. |
| `timings.csv` | Raw SG-ACO timing outputs. |
| `treeshap_timing.csv` | Aggregated TreeSHAP refresh-cost table. |
| `fig_gap.png` | Percentage gap to the best-known solution by instance and method. |
| `fig_conv.png` | Mean convergence curves for FT10 and TA01. |
| `fig_weights.png` | Signed TreeSHAP feature weights by instance. |

## Benchmark instances

The main experiments use ten named public JSSP benchmark instances with published best-known or optimal makespans:

`FT06`, `LA01`, `LA02`, `LA03`, `LA04`, `LA05`, `FT10`, `ABZ5`, `ORB01`, and `TA01`.

Instance metadata are stored in `instances.json`. The expected raw instance files should be placed under:

```text
instances/<instance_name>
```

For example:

```text
instances/ft06
instances/la01
instances/ta01
```

The Fisher–Thompson FT06 instance is available from Beasley’s OR-Library. Taillard-format instances should be generated or stored using the fixed seeds and paths reported in the study so that the experiments are exactly reproducible.

## Experimental setup

The stochastic methods are evaluated with the same budget:

- 25 independent seeds per stochastic method
- 12 ants × 250 iterations = 3000 schedule evaluations per ACO run
- 3000 neighbor evaluations for tabu search
- SG-ACO refresh interval `G = 25`
- ACO parameters: `alpha = 1`, `beta = 2`, `rho = 0.1`, `lambda = 1`

Compared methods:

- Standard ACO / Ant System (`AS-std`)
- MAX-MIN Ant System (`MMAS`)
- Ant Colony System (`ACS`)
- Tabu search (`Tabu`)
- Offline SHAP-guided ACO (`SHAP-off`)
- Closed-loop SG-ACO (`SG-ACO`)
- Deterministic rules: `SPT`, `MWKR`, and `MOR`

Because the benchmark instances optimize makespan and do not provide native due dates, the `JobSlack` feature uses the total-work-content rule:

```text
d_j = 1.5 × sum of processing times of job j
```

## Key figures

### Percentage gap to best-known solution

![Percentage gap to best-known solution](fig_gap.png)

### Convergence on FT10 and TA01

![Convergence curves](fig_conv.png)

### Signed TreeSHAP feature weights

![Signed TreeSHAP weights](fig_weights.png)

## Summary of main results

Average percentage gap to the best-known solution across the ten main instances:

| Method | Mean gap (%) |
|---|---:|
| Standard ACO (AS) | 44.37 |
| MAX-MIN Ant System | 40.83 |
| Ant Colony System | 47.32 |
| Tabu search | 28.84 |
| SHAP-ACO (offline) | 40.65 |
| SG-ACO (closed loop) | 39.55 |

The Friedman test over 250 BKS-normalized paired blocks rejects equal performance across the six stochastic methods:

```text
chi-square = 187.71
p-value    = 1.21e-38
blocks     = 250
```

Average Friedman ranks, where lower is better:

| Method | Average rank |
|---|---:|
| Tabu search | 2.394 |
| SG-ACO (closed loop) | 3.180 |
| SHAP-ACO (offline) | 3.378 |
| MAX-MIN Ant System | 3.620 |
| Ant Colony System | 3.888 |
| Standard ACO (AS) | 4.540 |

These results support the narrower claim that signed explanation-derived guidance improves the ACO family under the tested budget, while tabu search remains the strongest baseline overall on the larger instances.

## Installation

Python 3.12 is recommended.

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
```

Install dependencies:

```bash
pip install numpy pandas scipy scikit-posthocs matplotlib numba scikit-learn shap
```

Numba compiles several inner loops during the first run, so the first execution may be slower.

## Reproducing the experiments

1. Place the benchmark instance files in the `instances/` directory.
2. Run the resumable experiment grid:

```bash
python run_resumable.py
```

`run_resumable.py` is designed to be rerun safely. It appends completed runs to `results.csv` and exits periodically, so repeat the command until it exits with code `0`.

3. Generate summaries, statistical tests, and figures:

```bash
python analyze.py
```

4. Generate the manuscript-ready report:

```bash
python make_report.py
```

Expected generated outputs include:

```text
summary.csv
wilcoxon.csv
wilcoxon_pooled.csv
friedman.json
nemenyi.csv
ablation.csv
treeshap_timing.csv
fig_gap.png
fig_conv.png
fig_weights.png
report.md
```

If `fig_conv.png` is regenerated, the convergence-curve chunks `curves_*.npz` must be present. They are produced automatically by `run_resumable.py`.

## Data and reproducibility notes

The repository is intended to archive the complete computational record for the study:

- source code
- benchmark metadata
- instance files
- per-run results
- statistical outputs
- generated figures

For submission, update this section with the final public links:

```text
GitHub repository: <insert URL>
Zenodo DOI: <insert DOI>
```

Suggested data-availability statement:

> The FT06 benchmark instance is publicly available in Beasley’s OR-Library. Larger job-shop instances were generated using the standard Taillard procedure with the fixed random seeds reported in the paper. All experiment code, instance files, per-run results, and statistical outputs are included in this repository and archived with a Zenodo DOI.

## Suggested GitHub upload checklist

Before pushing the repository, confirm that the following are included:

```text
README.md
*.py
*.csv
*.json
*.png
instances/
curves_*.npz        # optional, needed to regenerate convergence curves exactly
inst_cache/         # optional, speeds reruns if cached surrogate artifacts are shared
```

Suggested commands:

```bash
git init
git add README.md *.py *.csv *.json *.png instances/
git commit -m "Add SG-ACO experiment archive"
git branch -M main
git remote add origin <your-github-repository-url>
git push -u origin main
```

## License

Add the license selected for the project before public release. For academic code releases, common choices are MIT, BSD-3-Clause, or Apache-2.0.

## Citation

Please cite the associated paper and archived release once the repository is public:

```bibtex
@misc{sgaco_experiments,
  title  = {SG-ACO Experiments: SHAP-Guided Ant Colony Optimization for Job-Shop Scheduling},
  author = {<insert authors>},
  year   = {<insert year>},
  doi    = {<insert Zenodo DOI>},
  url    = {<insert GitHub URL>}
}
```
