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

## plot_crashed_cycles

The set of scripts in this file plots different ways of analysing crashed simulations with the aim of gaining insight into the reasons.