#!/usr/bin/env python3
"""
Readers and converters for single-cell (CARP bench) simulation output.

These functions are vendored from GSA_library (M. Strocchi), which is not
publicly available, so that this repository can be cloned and run without it:

  - read_ionic_output            from GSA_library.file_utils
  - bin_to_dat, bin_to_dat_folder from GSA_library.ionic_output
  - plot_Land_output             from GSA_library.plotting

Only the functions used by this repository are included. The behaviour is
unchanged; the indentation has been normalised to 4 spaces.

A bench trace is stored as a text file of the form

    (99000): 0.115298,
    (99001): 0.115257,

where the number in brackets is the time in ms and the value after it is the
trace value.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import re


def read_ionic_output(filename):
    """Read a CARP cell-model trace file into a 1D array."""
    if not os.path.exists(filename):
        raise Exception('Cannot find ' + filename + '.')

    f = open(filename)
    lines = f.read().splitlines()

    data = []
    for i, line in enumerate(lines):
        sep = re.split(': |,|\n', line)

        if i == len(lines) - 1:
            data.append(float(sep[-1]))
        elif i > 0:
            data.append(float(sep[-2]))

    return np.array(data)


def bin_to_dat(filename,
               BCL,
               Nbeats,
               output_filename,
               cleanup=False,
               save_Nbeats=None):
    """Convert a bench .bin trace to the .dat text format, keeping the last beats."""
    if not os.path.exists(filename):
        raise Exception('Cannot find ' + filename + '.')

    s = np.fromfile(filename, dtype="float64")

    if save_Nbeats is None:
        save_Nbeats = Nbeats

    if s.shape[0] == save_Nbeats * BCL + 2:
        f = open(output_filename, "w")
        f.write("\n")

        for i in range(BCL * (save_Nbeats - 1), BCL * save_Nbeats - 1):
            f.write("      (" + str(i + (Nbeats - save_Nbeats) * BCL) + "): " + str(s[i]) + ",\n")
        f.write("      (" + str(int(BCL * Nbeats - 1)) + "): " + str(s[-2]))
        f.close()

    if cleanup:
        os.system("rm " + filename)


def bin_to_dat_folder(sim_foldername,
                      start_sample,
                      last_sample,
                      BCL,
                      Nbeats,
                      bin_files,
                      output_files,
                      cleanup=False,
                      save_Nbeats=None):
    """Run bin_to_dat over a range of numbered simulation subfolders."""
    for i in range(start_sample, last_sample + 1):

        folder = sim_foldername + "/" + str(i) + "/"

        for j, f in enumerate(bin_files):

            filename = folder + f
            bin_to_dat(filename,
                       BCL,
                       Nbeats,
                       folder + output_files[j],
                       cleanup=cleanup,
                       save_Nbeats=save_Nbeats)

        if cleanup:
            os.system("rm " + folder + "*.bin")


def plot_Land_output(basefolder,
                     N,
                     figname=None,
                     isometric=False,
                     figsize=(10, 5),
                     mask=[],
                     default='',
                     color='#3489eb'):
    """
    Plot the calcium transient, active tension, and stretch of Land cell simulations.

    Args:
        - basefolder: folder containing all simulations, numbered from 0 to N-1
        - N: number of simulations to plot
        - figname: path to the output figure. If None, the plot is shown
        - isometric: if True, no stretch is plotted
        - figsize: tuple, (width, height)
        - mask: boolean mask of which simulations terminated successfully
        - default: path to a folder with Land output to plot for comparison
    """
    if not isometric:
        ax = plt.figure(figsize=figsize, constrained_layout=True).subplots(1, 3)
    else:
        ax = plt.figure(figsize=figsize, constrained_layout=True).subplots(1, 2)

    plot_all = np.arange(N)
    if len(mask) > 0:
        plot_idx = plot_all[np.where(mask == 1)[0]]
    else:
        plot_idx = plot_all

    for i in plot_all:
        if not os.path.exists(basefolder + '/' + str(i) + '/Tension.dat'):
            raise Exception('Cannot find output file. The folder structure needs to be basefolder/i/Tension.dat and Ca_i.dat')

        T = read_ionic_output(basefolder + '/' + str(i) + '/Tension.dat')
        Ca_i = read_ionic_output(basefolder + '/' + str(i) + '/Ca_i.dat')
        t = np.arange(0, Ca_i.shape[0])

        if i in plot_idx:
            if np.max(np.abs(T)) < 500.0:
                ax[0].plot(t, Ca_i, color=color)
                ax[1].plot(t, T, color=color)
        else:
            if np.max(np.abs(T)) < 500.0:
                ax[0].plot(t, Ca_i, color='black', zorder=0)
                ax[1].plot(t, T, color='black', zorder=0)

        ax[0].set_xlabel('Time [ms]')
        ax[1].set_xlabel('Time [ms]')

        ax[0].set_ylabel('Ca_i [um]')
        ax[1].set_ylabel('Tension [kPa]')

        if not isometric:
            lambda_out = read_ionic_output(basefolder + '/' + str(i) + '/stretch.dat')
            if i in plot_idx:
                ax[2].plot(t, lambda_out, color=color)
            else:
                ax[2].plot(t, lambda_out, color='black', zorder=0)
            ax[2].set_xlabel('Time [ms]')
            ax[2].set_ylabel('Lambda [-]')

        if default != '':
            T = read_ionic_output(default + '/Tension.dat')
            Ca_i = read_ionic_output(default + '/Ca_i.dat')

            ax[0].plot(t, Ca_i, '--',
                       color='black',
                       linewidth=2.0
                       )
            ax[1].plot(t, T, '--',
                       color='black',
                       linewidth=2.0
                       )

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname, dpi=100)
