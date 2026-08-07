# simulation_toolbox

Scripts to set up, run, and post-process four-chamber cardiac electromechanical
simulations, and to build Gaussian process emulators and perform global sensitivity
analysis (GSA) on their outputs. The code was developed for a study of hypertrophic
cardiomyopathy (HCM) anatomies, including virtual drug scenarios (mavacamten and
aficamten).

The simulations themselves run with CARPentry (single-cell simulations with `bench`,
whole-heart jobs submitted through Slurm on ARCHER2 or the Imperial HPC). This
repository generates the simulation scripts, transfers files to and from the clusters,
checks and post-processes the results, and produces the analysis figures.

## Repository structure

```
notebooks/            Pipeline notebooks (numbered in running order) and utilities
simulation_toolbox/   Command-line scripts
├── common/           Shared helper functions (mesh I/O, GSA ranking files)
└── GPErks_modified/  Modified copy of GPErks for emulator training and GSA
```

`GPErks_modified` is a lightly modified copy of [GPErks](https://github.com/stelong/GPErks)
by Stefano Longobardi, included here for reproducibility. Credit for the original
implementation belongs to its author.

## Installation

- Install [pyenv](https://github.com/pyenv/pyenv) to handle Python versions, then
  create and activate a virtual environment:

  ```bash
  pyenv virtualenv 3.9.13 venv_simulation_toolbox
  pyenv local venv_simulation_toolbox
  ```

- Install the dependencies with [poetry](https://python-poetry.org/):

  ```bash
  poetry install
  ```

  `pyproject.toml` pulls `SIMULATION_library` and `UNLOADING_library`
  (M. Strocchi) directly from GitHub; you need access to those repositories. Some
  scripts also import [`GSA_library`](https://github.com/MarinaStrocchi/GSA_library),
  which is not part of `pyproject.toml`: clone it and install it into the same
  environment with `pip install <path_to_GSA_library>`.

- On a machine without a graphical interface (ARCHER2 or any headless server), pyvista
  needs the OSMesa VTK wheels:

  ```bash
  pip install vtk --extra-index-url https://wheels.vtk.org trame vtk-osmesa
  ```

## Data layout

The scripts assume one folder per anatomy and scenario, for example
`HCM/1/scenarios/53_more_samples/`, containing:

```
data/        X.txt (sampled parameters, one simulation per row), xlabels.txt,
             per-model parameter files (X_EP.txt, X_ToRORd_land.txt, ...), ylabels.txt
json_files/  one JSON parameter file per simulation
slrm/        Slurm submission scripts (unloading_<i>.slrm, cycle_<i>.slrm)
SS/          single-cell steady-state files (.sv), one subfolder per ionic model
             and simulation
output/      post-processed results (Y.txt, output masks, GSA_* subfolders)
```

Simulations are numbered by their row in `data/X.txt`, and scripts state explicitly
whether an index refers to the full sample or to the mask of working simulations.
Default paths in the scripts point to the authors' machines and external drives;
every script has an `argparse` interface, so run it with `--help` and override the
paths for your own setup.

## Simulation pipeline

1. Sample the parameter space with `notebooks/0_sampling.ipynb` (Latin hypercube
   design from the parameter ranges). `notebooks/reduce_intervals.ipynb` can shrink
   the ranges to the smallest interval containing all working simulations.
2. Generate the unloading Slurm scripts with
   `notebooks/1_generate_unloading_scripts.ipynb` (wraps `write_unloading_scripts.py`).
3. On the cluster, run `check_directories_and_files.sh` to make sure no file is
   missing, then submit the unloading jobs.
4. Check which unloadings converged with `check_unloading_convergence_archer2.py`
   (needs at least two simulations).
5. Run `screenshot_unloading_archer2.py` to take screenshots of the unloaded
   configurations. This script also prepares the folder structure for the cycle
   simulations, and can be run for that purpose alone.
6. Scale the electrophysiology and generate the cycle simulation scripts with
   `notebooks/2_scale_EP_generate_cycle.ipynb` (uses `write_simulation_scripts.py` and
   `patient_specific_CV.py`).
7. Run the single-cell simulations to steady state with
   `notebooks/3_run_cell_simulations.ipynb` (uses `bench`); the resulting `.sv` state
   files populate `SS/`. `notebooks/3_run_from_samples.ipynb` covers the case where new
   simulations are generated from an existing sample. `generate_bench_script_skip_overwrite.py`
   re-generates bench scripts without overwriting completed runs.
8. Transfer the Slurm scripts and `.sv` files to the cluster with `scp_files_hpc.sh`.
   `scp_unloading.sh` and `unpack_imperial.sh` move unloaded meshes from ARCHER2 to the
   Imperial HPC. Set the username placeholders at the top of these scripts first.
9. Repeat step 3 for the cycle simulations and submit them.
10. Post-process the finished cycles with `check_cycle_output_archer2.py`, which
    extracts the mechanics outputs (`Y.txt`) and the mask of working simulations.
    `clean_unloading_cycle.sh` tidies up the simulation folders on the cluster. To
    investigate crashed simulations, use `plot_crashed_cycles.py`.
11. Optional variants: `notebooks/4_generate_inflation_script.ipynb` with
    `write_inflation_scripts.py` and `check_inflation_output_archer2.py` for inflation
    experiments, and `notebooks/5_EP_simulations.ipynb` with `run_EP_simulations.py`,
    `check_cycle_EP.py`, and `check_cycle_motion_archer2.py` for electrophysiology-only
    runs. `notebooks/default.ipynb` and `notebooks/default multimesh.ipynb` run the
    default (unsampled) unloading for one or several meshes.

## Emulation and GSA

- `notebooks/6_emulation.ipynb` and `train_GPEs.py` train one Gaussian process
  emulator per output using `GPErks_modified`.
- `notebooks/7_GSA.ipynb` runs the global sensitivity analysis on the trained
  emulators; `notebooks/plot_GSA.ipynb` draws radar charts comparing GSAs.
- `compare_gsa.py` compares GSA results across scenarios or anatomies, and
  `create_emulator_metric_summary_table.py` summarises emulator accuracy metrics.

## Analysis and figure scripts

Most figure scripts have a matching `run_*.sh` wrapper that records the exact
arguments used for the study; start from the wrapper and adapt the paths.

| Script | Description |
|---|---|
| `plot_sensitivity_boxplot.py` | Boxplots of GSA sensitivities across anatomies |
| `plot_sensitivity_boxplot_comparison.py` | Sensitivity boxplots comparing baseline and modified parameter intervals |
| `plot_sensitivity_boxplot_pharma.py` | Sensitivity boxplots for the drug scenarios |
| `plot_sensitivity_scatter.py` | Scatter plots of sensitivities between scenarios |
| `plot_baseline_sensitivity_summary.py` | Summary figure of baseline sensitivities |
| `plot_ranking_heatmap.py` | Heatmap of parameter importance rankings |
| `plot_ranking_difference_heatmap.py` | Ranking changes after modifying parameter intervals (`plot_modified_intervals.sh`) |
| `plot_ranking_variability_heatmap.py` | Ranking variability across anatomies (`plot_ranking_variability_heatmap_pairs.sh`) |
| `plot_ranking_alluvial.py` | Alluvial diagram of ranking changes |
| `plot_bump_chart.py` | Bump chart of ranking changes |
| `plot_sensitivities_outputs.py` | Sensitivities alongside output distributions |
| `plot_vas_comparison.py` | Variability in Anatomical Sensitivity (VAS) comparison |
| `print_comparison_analysis.py` | Printed statistics comparing scenarios |
| `print_comparison_functional_vs_anatomical.py` | Functional versus anatomical contribution statistics |
| `print_variability.py` | Printed ranking variability statistics |
| `generate_csv_drugs_effect.py` | CSV tables of the simulated drug effects |
| `validate_drug_cell_sweep.py` | Single-cell validation of the drug parameter mappings (see below) |
| `plot_HCM_cell_traces_drugs.py` | Single-cell tension and calcium traces with drug ranges highlighted |
| `plot_HCM_pv_loops.py` | PV loops of all beats and chambers, one figure per HCM case |
| `plot_HCM_all_cycles.py` | Pressure and volume traces of all cycles per case |
| `compute_beat4_vs_beat5_stats.py` | Convergence statistics of the outputs between beats 4 and 5 |
| `plot_graphical_abstract_panels.py` | Panels for the graphical abstract |
| `plot_output.py` | General plots of simulation outputs |
| `plot_output_distributions.py` | Distributions of the simulation outputs |
| `plot_PV_loops.py` | PV loops of a single simulation |
| `visualise_motion.py`, `visualise_motion_sliced.py` | 3D rendering of the simulated motion |
| `video_heartbeat_pvloop.py` | Video combining the beating mesh and its PV loop |
| `visualise_EP.py` | Activation time maps |

## Drug cell sweep (`validate_drug_cell_sweep.py`)

Runs controlled single-cell isometric bench simulations to validate that the Land
active-contraction parameters used to model the drugs (mavacamten, aficamten) produce
the expected negative inotropy. Sweeps one contraction parameter at a time across its
full sampled range and reports peak and developed tension and transient overlays.

Prerequisites: a compiled `bench.pt` binary (CARPentry) and the scenario `data/`
folder of the anatomy to validate.

Full run (45 bench simulations, several minutes with 6 workers):

```bash
cd simulation_toolbox
python validate_drug_cell_sweep.py \
    --output_dir <output_folder> \
    --workdir <output_folder>/workdir \
    --keep_workdir \
    --workers 6
```

`--keep_workdir` preserves the bench outputs so you can re-plot without re-running
the simulations; use a persistent path for `--workdir` (not `/tmp`). To re-plot only,
pass `--replot` with the same `--workdir`. Outputs: `drug_cell_sweep.png` (dose
response and transient overlays) and `drug_cell_sweep_metrics.csv` (peak, developed,
and diastolic tension, RT50, RT90).

## Licence

This code is released under the MIT licence (see `LICENSE`). The
`GPErks_modified` directory contains modified code from
[GPErks](https://github.com/stelong/GPErks), also MIT licensed.

## Citation

If you use this code, please cite the associated publication.
[Citation to be added.]
