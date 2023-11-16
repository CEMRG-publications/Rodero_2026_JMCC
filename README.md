# Updated scripts to post process simulations (mainly mechanics)

## Install

The recommended installation is as follows:

- Install `pyenv` to handle different python versions.
- Create a virtual environment, for example with
 ```
 pyenv virtualenv 3.9.13 venv_simulation_toolbox
 ```
 - Activate it:
 ```
 pyenv local venv_simulation_toolbox
 ```
 - Install all the dependencies:
 ```
 poetry add requests
 ```
 - If running it in ARCHER2 or any other computer without a graphic interface, pyvista won't work. To fix it run:
 ```
 pip install vtk --extra-index-url https://wheels.vtk.org trame vtk-osmesa
 ```

 # Steps

 1. Sample the parameter space following the notebook `0_sampling.ipynb`
 2. Generate the unloading scripts following the notebook `1_generate_unloading_script.ipynb`
 3. In ARCHER2: Take a `.slrm` file and run `check_directories_and_files.sh` to make sure you are not missing any file.
 4. In ARCHER2: Check which unloadings work using `check_unloading_convergence_archer2.py`
 5. In ARCHER2: Take screenshots of the unloading configurations using `screenshot_unloading_archer2.py`. This script is necessary to prepare the folder structure for the cycle simulations. You can run it to prepare the folders without taking the screenshots.
 6. Run cell simulations and generate the cycle simulations following the notebook `2_run_cell_generate_cycle.ipynb`
 7. Repeat step 3. with the cycle simulation slrms.
 5. Run `check_cycle_output_archer2.py`.
5. If you want to analyse the crashed simulation, run `plot_crashed_cycles`. The set of scripts in this file plots different ways of analysing crashed simulations with the aim of gaining insight into the reasons.