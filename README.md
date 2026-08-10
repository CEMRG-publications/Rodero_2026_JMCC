# simulation_toolbox

Code for the paper:

> Rodero et al. (2026). Computational models show that functional remodeling,
> not anatomy, reshapes physiological determinants of cardiac performance in
> hypertrophic cardiomyopathy. *Journal of Molecular and Cellular Cardiology*.

The repository sets up, post-processes, and analyses four-chamber cardiac
electromechanical simulations of five hypertrophic cardiomyopathy (HCM)
anatomies, including virtual mavacamten and aficamten scenarios. It builds
Gaussian process emulators of the simulation outputs, runs a global sensitivity
analysis (GSA) on them, and produces the figures and tables of the paper.

The simulations themselves were run with CARPentry (single-cell simulations with
`bench`, whole-heart jobs submitted through Slurm on ARCHER2 and the Imperial
HPC). This repository generates those simulation scripts, moves files to and from
the clusters, checks and post-processes the results, and performs the analysis.

## Quick start

```bash
git clone https://github.com/crisrogo/simulation_toolbox.git
cd simulation_toolbox
poetry install
./reproduce_figures.sh
```

`reproduce_figures.sh` runs against the small dataset in `example_data/`, so it
works immediately after cloning with nothing to download, and writes its figures
and tables to `results/`. Point `DATA_ROOT` at a full Zenodo download to run the
same analysis over the complete output set.

## Data

The example dataset in `example_data/` (1.5 MB) holds the sensitivity indices and
parameter rankings for the five anatomies and the two drug scenarios. It is
enough to check an installation and to see the analysis end to end. See
`example_data/README.md` for its layout.

The full data are deposited on Zenodo. Each deposit is the output of one stage of
the pipeline below, so you can start at whichever stage you are interested in
rather than rerunning the ones before it:

| Dataset | Pipeline stage | DOI |
|---|---|---|
| Meshes | Input anatomies | [10.5281/zenodo.21282274](https://doi.org/10.5281/zenodo.21282274) |
| Electrophysiology simulations | Step 11 | [10.5281/zenodo.21720235](https://doi.org/10.5281/zenodo.21720235) |
| Inflation simulations | Step 11 | [10.5281/zenodo.21720634](https://doi.org/10.5281/zenodo.21720634) |
| Unloading simulations | Steps 2 to 5 | [10.5281/zenodo.21809471](https://doi.org/10.5281/zenodo.21809471) |
| Cycle simulations | Steps 6 to 9 | [10.5281/zenodo.21822518](https://doi.org/10.5281/zenodo.21822518) |

The cycle simulations are the ones the paper's results are computed from. With
that deposit you can rerun the whole post-processing chain, from the raw
CARPentry output to `Y.txt` and on to every figure, using only this repository.

### Where the scripts look for data

Every script and wrapper reads two environment variables rather than any
hardcoded path:

```bash
export DATA_ROOT=/path/to/data       # holds HCM/<case>/scenarios/<scenario>/
export RESULTS_ROOT=/path/to/output  # defaults to $DATA_ROOT/results
```

A scenario folder is laid out as:

```
data/        X.txt (sampled parameters, one simulation per row), xlabels.txt,
             ylabels.txt, per-model parameter files (X_EP.txt, X_ToRORd_land.txt, ...)
json_files/  one JSON parameter file per simulation
slrm/        Slurm submission scripts (unloading_<i>.slrm, cycle_<i>.slrm)
SS/          single-cell steady-state files (.sv), per ionic model and simulation
simulations/ raw CARPentry output, one folder per cycle
output/      Y.txt, output masks, Si_total.csv, Rank_* files, GSA_* subfolders
```

Simulations are numbered by their row in `data/X.txt`. Scripts state explicitly
whether an index refers to the full sample or to the mask of working simulations.
A few scripts accept several roots at once, colon-separated like `PATH`, because
the study data were spread over more than one drive.

## Installation

Python 3.9.13 is required. The recommended setup uses
[pyenv](https://github.com/pyenv/pyenv) and [poetry](https://python-poetry.org/):

```bash
pyenv install 3.9.13
pyenv virtualenv 3.9.13 venv_simulation_toolbox
pyenv local venv_simulation_toolbox
poetry install
```

`poetry install` gives everything needed to reproduce the analysis and the paper
figures. Two optional extras cover the rest:

```bash
poetry install --extras emulation      # train emulators and run the GSA
poetry install --extras visualisation  # 3D mesh rendering with pyvista/vtk
```

Every dependency resolves from PyPI. Nothing in this repository depends on a
private package.

On a machine without a graphical interface, pyvista needs the OSMesa VTK wheels:

```bash
pip install vtk --extra-index-url https://wheels.vtk.org trame vtk-osmesa
```

### Running new simulations

The Python code here is entirely public. Generating and post-processing
simulations needs no private package.

Running a simulation is a separate matter. The scripts produce Slurm submission
files that invoke CARPentry and its `bench` single-cell solver, which are
licensed separately and are not distributed here, and the jobs need an HPC
allocation. Everything up to the submission file, and everything from the raw
output onwards, works with a plain `poetry install`.

One step has a further requirement. The unloading job generated by
`write_unloading_scripts.py` runs `run.py` from `UNLOADING_library`, which has to
be installed on the cluster and is not public: pass its location with
`--HPC_path_to_unloading_library`. This affects only the generation of new
unloaded meshes. The unloaded configurations used in the paper are on Zenodo, so
reproducing the published results does not require it.

## Reproducing the paper figures

```bash
./reproduce_figures.sh
```

runs the drug-scenario analysis: variability in anatomical sensitivity (VAS)
across the five anatomies, the sensitivity boxplots, and the printed comparison
statistics, for each chamber group. On the example data this takes under a minute
and writes 11 figures and 54 tables to `results/`.

The individual analyses can also be run one at a time. Each figure script has a
matching `run_*.sh` wrapper recording the exact arguments used for the study, so
start from the wrapper:

```bash
cd simulation_toolbox
export DATA_ROOT=/path/to/data
bash run_plot_VAS_comparison_pharma.sh                    # drug scenarios
bash run_plot_VAS_comparison_functional_remodeling.sh     # functional remodelling
bash run_plot_sensitivity_boxplot.sh                      # baseline sensitivities
bash run_plot_ranking_heatmap.sh                          # parameter rankings
```

Scripts that consume parameter rankings read `Rank_Si_total_max_<output>.txt`
files. The example data ship those files. To regenerate them from a
`Si_total.csv` matrix for new data, use `generate_ranking_files.py`:

```bash
python generate_ranking_files.py \
    --xlabels      $DATA_ROOT/HCM/1/scenarios/53_more_samples/data/xlabels.txt \
    --ylabels      $DATA_ROOT/HCM/1/scenarios/53_more_samples/data/ylabels.txt \
    --ylabels_dict $DATA_ROOT/HCM/GSA_analysis/cycle/ylabels_filtered.json \
    --scenarios    $DATA_ROOT/HCM/1/scenarios/53_more_samples/output
```

Every script has an `argparse` interface, so run it with `--help` to see the
options and defaults.

## Repository structure

```
example_data/         Small dataset so the analysis runs straight after cloning
notebooks/            Pipeline notebooks, numbered in running order
reproduce_figures.sh  One command from a clean clone to the drug-scenario figures
simulation_toolbox/   Command-line scripts
├── common/           Shared helpers: mesh and cell I/O, GSA ranking files
│   └── carp_setup/   Simulation-setup code for writing CARPentry submissions
└── GPErks_modified/  Modified copy of GPErks, for emulator training and GSA
```

## Simulation pipeline

How the study data were produced, and how to produce more. Every step below runs
with a plain `poetry install`; the simulations the scripts submit need CARPentry
and an HPC allocation, which you obtain separately.

You do not have to start at step 1. The output of each stage is on Zenodo, so to
reproduce the published results you can download the cycle simulations and go
straight to step 10.

1. Sample the parameter space with `notebooks/0_sampling.ipynb` (Latin hypercube
   design). `notebooks/reduce_intervals.ipynb` shrinks the ranges to the smallest
   interval containing all working simulations.
2. Generate the unloading Slurm scripts with
   `notebooks/1_generate_unloading_scripts.ipynb`, which wraps
   `write_unloading_scripts.py`.
3. On the cluster, run `check_directories_and_files.sh` to confirm no file is
   missing, then submit the unloading jobs.
4. Check which unloadings converged with
   `check_unloading_convergence_archer2.py`, which needs at least two simulations.
5. Run `screenshot_unloading_archer2.py` to render the unloaded configurations.
   It also prepares the folder structure for the cycle simulations, and can be
   run for that alone.
6. Scale the electrophysiology and generate the cycle scripts with
   `notebooks/2_scale_EP_generate_cycle.ipynb`, which uses
   `write_simulation_scripts.py` and `patient_specific_CV.py`.
7. Run the single-cell simulations to steady state with
   `notebooks/3_run_cell_simulations.ipynb`; the resulting `.sv` files populate
   `SS/`. `notebooks/3_run_from_samples.ipynb` covers generating new simulations
   from an existing sample, and `generate_bench_script_skip_overwrite.py`
   regenerates bench scripts without overwriting completed runs.
8. Transfer the Slurm scripts and `.sv` files with `scp_files_hpc.sh`.
   `scp_unloading.sh` and `unpack_imperial.sh` move unloaded meshes from ARCHER2
   to the Imperial HPC. Set the username placeholders at the top of these scripts
   first.
9. Repeat step 3 for the cycle simulations and submit them.
10. Post-process the finished cycles with `check_cycle_output_archer2.py`, which
    extracts the mechanics outputs into `Y.txt` together with the mask of working
    simulations. This is the entry point if you start from the cycle simulations
    downloaded from Zenodo. `clean_unloading_cycle.sh` tidies the cluster folders, and
    `plot_crashed_cycles.py` investigates simulations that failed.
11. Optional variants: `notebooks/4_generate_inflation_script.ipynb` with
    `write_inflation_scripts.py` and `check_inflation_output_archer2.py` for
    inflation experiments; `notebooks/5_EP_simulations.ipynb` with
    `run_EP_simulations.py`, `check_cycle_EP.py`, and
    `check_cycle_motion_archer2.py` for electrophysiology-only runs;
    `notebooks/default.ipynb` and `notebooks/default multimesh.ipynb` for the
    default, unsampled unloading of one or several meshes.

## Emulation and GSA

`notebooks/6_emulation.ipynb` and `train_GPEs.py` train one Gaussian process
emulator per output using `GPErks_modified`. `notebooks/7_GSA.ipynb` runs the
global sensitivity analysis on the trained emulators, and
`notebooks/plot_GSA.ipynb` draws radar charts comparing analyses. `compare_gsa.py`
compares results across scenarios or anatomies, and
`create_emulator_metric_summary_table.py` summarises emulator accuracy.

## Analysis and figure scripts

| Script | Description |
|---|---|
| `plot_vas_comparison.py` | Variability in anatomical sensitivity (VAS) across anatomies |
| `plot_sensitivity_boxplot.py` | Boxplots of GSA sensitivities across anatomies |
| `plot_sensitivity_boxplot_comparison.py` | Sensitivities for baseline against modified parameter intervals |
| `plot_sensitivity_boxplot_pharma.py` | Sensitivities for the drug scenarios |
| `plot_sensitivity_scatter.py` | Scatter plots of sensitivities between scenarios |
| `plot_baseline_sensitivity_summary.py` | Summary figure of the baseline sensitivities |
| `plot_ranking_heatmap.py` | Heatmap of parameter importance rankings |
| `plot_ranking_difference_heatmap.py` | Ranking changes after modifying parameter intervals |
| `plot_ranking_variability_heatmap.py` | Ranking variability across anatomies |
| `plot_ranking_alluvial.py` | Alluvial diagram of ranking changes |
| `plot_bump_chart.py` | Bump chart of ranking changes |
| `plot_sensitivities_outputs.py` | Sensitivities alongside output distributions |
| `print_comparison_analysis.py` | Printed statistics comparing scenarios |
| `print_comparison_functional_vs_anatomical.py` | Functional against anatomical contribution |
| `print_variability.py` | Printed ranking variability statistics |
| `generate_csv_drugs_effect.py` | CSV tables of the simulated drug effects |
| `generate_ranking_files.py` | Build the ranking files from a `Si_total.csv` matrix |
| `validate_drug_cell_sweep.py` | Single-cell validation of the drug parameter mappings |
| `plot_HCM_cell_traces_drugs.py` | Cell tension and calcium traces, drug ranges highlighted |
| `plot_HCM_pv_loops.py` | PV loops of all beats and chambers, one figure per case |
| `plot_HCM_all_cycles.py` | Pressure and volume traces of all working cycles |
| `compute_beat4_vs_beat5_stats.py` | Convergence of the outputs between beats 4 and 5 |
| `plot_graphical_abstract_panels.py` | Panels for the graphical abstract |
| `plot_output.py`, `plot_output_distributions.py` | Quality checks on the simulation outputs |
| `plot_PV_loops.py` | PV loops of a single simulation |
| `visualise_motion.py`, `visualise_motion_sliced.py` | 3D rendering of the simulated motion |
| `video_heartbeat_pvloop.py` | Video combining the beating mesh with its PV loop |
| `visualise_EP.py` | Activation time maps |

### Single-cell drug validation

`validate_drug_cell_sweep.py` runs controlled single-cell isometric bench
simulations to check that the Land active-contraction parameters used to model
mavacamten and aficamten produce the expected negative inotropy. It sweeps one
contraction parameter at a time across its sampled range and reports peak and
developed tension with the transients overlaid. It needs a compiled `bench.pt`
(set the `BENCH` environment variable to its path) and the `data/` folder of the
scenario being validated:

```bash
export DATA_ROOT=/path/to/data
export BENCH=/path/to/bench.pt
python validate_drug_cell_sweep.py --output_dir $RESULTS_ROOT/drug_validation \
                                   --workdir    $RESULTS_ROOT/drug_validation/workdir \
                                   --keep_workdir --workers 6
```

`--keep_workdir` preserves the bench output so the figure can be redrawn with
`--replot` without rerunning the simulations. Use a persistent path for
`--workdir`, not `/tmp`.

## Vendored code

`GSA_library`, `SIMULATION_library`, `UNLOADING_library`, and `DATA_library` by
M. Strocchi are not publicly available, so the code this repository needs from
them is included here rather than imported. Credit for that code belongs to its
author.

`common/mesh_io.py`, `common/cell_io.py`, and `common/fourchamber_output.py` hold
the file readers and the derived haemodynamic measures used by the analysis. Only
the functions used are included, with the indentation normalised; each was checked
to give numerically identical results to the original.

`common/carp_setup/` holds the simulation-setup code: the `simulation`
configuration class, the writers that turn a parameter set into a CARPentry
submission script, the single-cell script generator, and the ARCHER2 job headers.
These modules are verbatim copies, tab indentation included, with only their
internal imports repointed, so that the files they generate stay byte-identical
to the ones used for the published simulations. Every one of their 37 top-level
definitions was diffed against the original to confirm this.

`simulation_toolbox/GPErks_modified/` is a modified copy of
[GPErks](https://github.com/stelong/GPErks) by Stefano Longobardi, included for
reproducibility. Credit for the original implementation belongs to its author.

## Licence

MIT, see `LICENSE`. The vendored `GPErks_modified` code is also MIT licensed.

## Citation

If you use this code, please cite the paper above; `CITATION.cff` holds the
machine-readable version. If you use the data, please also cite the relevant
Zenodo deposit.
