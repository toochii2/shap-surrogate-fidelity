# SHAP explanations of building-simulation surrogates: code and data

Code and simulation results for the manuscript "Accurate but misleading: testing explainable AI for building-simulation surrogates against EnergyPlus ground truth".

Author: Ali Alsayegh, Department of Design, Manufacturing and Engineering Management, University of Strathclyde, Glasgow, United Kingdom (a.alsayegh@strath.ac.uk, ORCID [0000-0001-7083-3639](https://orcid.org/0000-0001-7083-3639)).

## Contents

```
code/      Python scripts (model, designs, analyses, figures)
results/   simulation results and analysis outputs (parquet, csv, pkl)
weather/   place the TMYx weather files here (not redistributed)
```

## Requirements

- EnergyPlus 26.2. Set the `ENERGYPLUS_EXE` environment variable to its executable, place it in `tools/EnergyPlus-26.2*/`, or have `energyplus` on the PATH.
- Python 3.12 with the packages in `requirements.txt`.
- The weather files listed in `weather/README.txt`.

The results are included, so the analysis and figure scripts run without EnergyPlus. Only the stages marked "EnergyPlus" below need it.

## Running

Run all scripts from `code/`. Each script takes one or more stage names as arguments.

| Step | Command | Paper |
|---|---|---|
| Main simulations (EnergyPlus) | `python run_designs.py saltelli train pdp local strong` | Sections 2.2–2.3 |
| First-pass scoring | `python xai.py` and `python xai2.py` | Sections 3.2–3.6 |
| Repeated training, Sobol indices, baselines, Eq. (8) | `python analysis_repeats.py sobol repeats baselines ovb` | Sections 3.1–3.6 |
| Correlation sweep, infiltration omission, retrofit (EnergyPlus for `sims`) | `python analysis_omission.py sims sweep samedir retrofit budget` | Section 3.7 |
| Supplementary checks (EnergyPlus for `extins`) | `python analysis_checks.py excess signs budget extins numbers` | Sections 3.1, 3.5–3.7 |
| Envelope inputs and detection matrix | `python analysis_detection.py envelope detect gamma ridge misc` | Sections 3.5, 3.7, 4.1; Tables S2–S3 |
| Second climate (EnergyPlus for `sims`) | `EP_CLIMATE=KHI EP_RESULTS_SUB=karachi python replication.py sims analysis signs retrofit summary` | Section 3.8 |
| Second building type (EnergyPlus for `sims`) | `EP_BTYPE=LIGHT EP_RESULTS_SUB=light python replication.py sims analysis signs retrofit summary` | Section 3.8 |
| Comparison of the three cases | `python replication_compare.py` | Table S4 |
| Figures | `python make_figs.py 1 2 5`, `python make_figs_fidelity.py`, `python make_figs_omission.py 9 t4` | Figures 1–8, S1; Table 2 |

The full EnergyPlus set (258,622 runs) takes about 3 h on 14 cores.

## Results files

| Prefix | Content |
|---|---|
| `saltelli`, `train_*`, `pdp_truth_*`, `local_truth_*`, `local_instances_*` | EnergyPlus runs: Sobol design, training designs, interventional main effects, baseline Shapley values |
| `xai_*`, `xai2_*` | First-pass fidelity scores |
| `p1_*` | Repeated training: accuracy, fidelity, per-input errors, slope ratios, Sobol indices, Eq. (8) |
| `p2_*` | Correlation sweep, infiltration omission and retrofit runs |
| `r2_*` | Excess-temperature Sobol indices, sign accuracy, budget check, external insulation |
| `r3_*` | Envelope-input fidelity, detection matrix, subsample estimates of Eq. (8) |
| `karachi/`, `light/` | Second climate and second building type |
| `khi_compare.csv` | Key results for the three cases |

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0.
