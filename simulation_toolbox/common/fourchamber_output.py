#!/usr/bin/env python3
"""
Derived haemodynamic measures for the atria of a four-chamber simulation.

These functions are vendored from SIMULATION_library.fourchamber_output
(M. Strocchi), which is not publicly available, so that the post-processing and
figure scripts in this repository can be run without it. Only the functions used
here are included. The behaviour is unchanged; the indentation has been
normalised to 4 spaces.

The two check_* functions draw diagnostic figures of how each measure was picked
off the pressure and volume traces; they are used by check_cycle_output_archer2.py
when it is run with plotting enabled.
"""

import matplotlib.pyplot as plt
import numpy as np


def AA_output(time, volume, pressure):
    """Return [EDV, ESV, maxV, maxP] of an atrium from its volume and pressure traces."""
    dv = np.gradient(volume)

    # Find the beginning of ejection
    idx_start_ej = np.where(dv < 0)[0][0]

    EDV = volume[idx_start_ej]
    # To make sure we pick the minimum during the a-wave
    ESV = np.min(volume[idx_start_ej:int(volume.shape[0] / 2)])

    end_awave = np.where(volume == ESV)[0][0]
    maxV = np.max(volume[end_awave:])
    maxP = np.max(pressure[:end_awave])

    return [EDV, ESV, maxV, maxP]


def AA_output_ej(time, volume):
    """Return [SV of the a-wave, inflow volume, SV of the v-wave] of an atrium."""
    dv = np.gradient(volume)

    # Find the beginning of ejection
    idx_start_ej = np.where(dv < 0)[0][0]

    EDV = volume[idx_start_ej]
    # To make sure we pick the minimum during the a-wave
    ESV = np.min(volume[idx_start_ej:int(volume.shape[0] / 2)])

    end_awave = np.where(volume == ESV)[0][0]
    maxV = np.max(volume[end_awave:])

    AA_SV_a = EDV - ESV
    AA_infl_v = maxV - EDV
    half_vwave = end_awave + np.where(volume[end_awave:] == maxV)[0][0]

    # Check that the volume goes back down, that is, that the atrium
    # ejects during the v-wave
    ESV_v = np.min(volume[half_vwave:])

    AA_SV_v = maxV - ESV_v

    return [AA_SV_a, AA_infl_v, AA_SV_v]


def check_ventricle_output(time, volume, pressure, measures):
    """Draw a diagnostic figure of how the ventricular measures were extracted."""
    colors = ['#ff9900', '#00cc00', '#ff3300', '#33ccff', '#ff66ff', '#6666ff']

    dpdt_ = np.diff(pressure) / np.diff(time) * 1000.0
    dpdt = np.zeros(pressure.shape, dtype=float)
    dpdt[1:] = dpdt_
    dpdtmax_time = np.where(dpdt == measures[4])[0]
    dpdtmin_time = np.where(dpdt == measures[5])[0]

    fig = plt.figure(figsize=(15, 10))
    pv_plt = fig.add_subplot(3, 2, (1, 3))
    p_plt = fig.add_subplot(3, 2, 2)
    dp_plt = fig.add_subplot(3, 2, 4)
    v_plt = fig.add_subplot(3, 2, 6)

    pv_plt.set_xlabel('Volume [mL]')
    pv_plt.set_ylabel('Pressure [mmHg]')
    pv_plt.plot(volume, pressure, color='k', linewidth=2.0)
    pv_plt.vlines(x=measures[0], ymin=np.min(pressure) - 2.0, ymax=np.max(pressure) + 2.0, linewidth=3, color=colors[0])
    pv_plt.text(measures[0], np.max(pressure) + 2.0, 'EDV', fontsize=22, color=colors[0])

    pv_plt.hlines(y=measures[1], xmin=np.min(volume) - 5.0, xmax=np.max(volume) + 5.0, linewidth=3, color=colors[2])
    pv_plt.text(np.max(volume) + 5.0, measures[1], 'EDP', fontsize=22, color=colors[2])

    pv_plt.vlines(x=measures[2], ymin=np.min(pressure) - 2.0, ymax=np.max(pressure) + 2.0, linewidth=3, color=colors[1])
    pv_plt.text(measures[2], np.max(pressure) + 2.0, 'ESV', fontsize=22, color=colors[1])

    pv_plt.hlines(y=measures[3], xmin=np.min(volume) - 5.0, xmax=np.max(volume) + 5.0, linewidth=3, color=colors[3])
    pv_plt.text(np.max(volume) + 5.0, measures[3], 'pMax', fontsize=22, color=colors[3])

    dp_plt.set_xlabel('Time [ms]')
    dp_plt.set_ylabel('dpdt [mmHg/s')
    dp_plt.plot(time, dpdt, color='k', linewidth=2.0)
    dp_plt.scatter(time[dpdtmax_time], measures[4] * np.ones(dpdtmax_time.shape), s=20.0, color=colors[4])
    dp_plt.scatter(time[dpdtmin_time], measures[5] * np.ones(dpdtmin_time.shape), s=20.0, color=colors[5])

    p_plt.set_xlabel('Time [ms]')
    p_plt.set_ylabel('Pressure [mmHg]')
    p_plt.plot(time, pressure, color='k', linewidth=2.0)
    p_plt.hlines(y=measures[1], xmin=time[0], xmax=time[-1], linewidth=3, color=colors[2])
    p_plt.hlines(y=measures[3], xmin=time[0], xmax=time[-1], linewidth=3, color=colors[3])
    p_plt.scatter(time[dpdtmax_time], pressure[dpdtmax_time], s=20.0, color=colors[4])
    p_plt.scatter(time[dpdtmin_time], pressure[dpdtmin_time], s=20.0, color=colors[5])

    v_plt.set_xlabel('Time [ms]')
    v_plt.set_ylabel('Volume [mL]')
    v_plt.plot(time, volume, color='k', linewidth=2.0)
    v_plt.hlines(y=measures[0], xmin=time[0], xmax=time[-1], linewidth=3, color=colors[0])
    v_plt.hlines(y=measures[2], xmin=time[0], xmax=time[-1], linewidth=3, color=colors[1])

    plt.show()


def check_atria_output(time, volume, pressure, measures):
    """Draw a diagnostic figure of how the atrial measures were extracted."""
    colors = ['#ff9900', '#00cc00', '#ff3300', '#33ccff', '#ff66ff', '#6666ff']

    fig = plt.figure(figsize=(15, 10))
    pv_plt = fig.add_subplot(2, 2, (1, 3))
    p_plt = fig.add_subplot(2, 2, 2)
    v_plt = fig.add_subplot(2, 2, 4)

    pv_plt.set_xlabel('Volume [mL]')
    pv_plt.set_ylabel('Pressure [mmHg]')
    pv_plt.plot(volume, pressure, color='k', linewidth=2.0)
    pv_plt.vlines(x=measures[0], ymin=np.min(pressure) - 2.0, ymax=np.max(pressure) + 2.0, linewidth=3, color=colors[0])
    pv_plt.text(measures[0], np.max(pressure) + 2.0, 'EDV', fontsize=22, color=colors[0])

    pv_plt.vlines(x=measures[1], ymin=np.min(pressure) - 2.0, ymax=np.max(pressure) + 2.0, linewidth=3, color=colors[1])
    pv_plt.text(measures[1], np.max(pressure) + 2.0, 'ESV', fontsize=22, color=colors[1])

    pv_plt.vlines(x=measures[2], ymin=np.min(pressure) - 2.0, ymax=np.max(pressure) + 2.0, linewidth=3, color=colors[2])
    pv_plt.text(measures[2], np.max(pressure) + 2.0, 'maxV', fontsize=22, color=colors[1])

    p_plt.set_xlabel('Time [ms]')
    p_plt.set_ylabel('Pressure [mmHg]')
    p_plt.plot(time, pressure, color='k', linewidth=2.0)

    v_plt.set_xlabel('Time [ms]')
    v_plt.set_ylabel('Volume [mL]')
    v_plt.plot(time, volume, color='k', linewidth=2.0)
    v_plt.hlines(y=measures[0], xmin=time[0], xmax=time[-1], linewidth=3, color=colors[0])
    v_plt.hlines(y=measures[1], xmin=time[0], xmax=time[-1], linewidth=3, color=colors[1])
    v_plt.hlines(y=measures[2], xmin=time[0], xmax=time[-1], linewidth=3, color=colors[2])

    plt.show()
