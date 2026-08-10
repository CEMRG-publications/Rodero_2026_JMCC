import json
from pandas import read_csv
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tqdm


def plot_biomarkers_VV(simulation_array,datafolder,mask_file,NBEATS,BCL,AVD,sims_folder,fig_path,output_file,chamber):

    print(f"Plotting biomarkers for the {chamber}...")

    
    mask = np.loadtxt(mask_file,dtype=int)

    ylabels = np.loadtxt(f"{datafolder}/ylabels.txt", dtype="str")

    dict_labels = {"LV_volume": ["LVedv","LVesv"],
                   "LV_pressure": ["LVedp","LVpMax"],
                   "LV_dpressure": ["LVdpdtMax","LVdpdtMin"],
                   "LV_ratio": ["LVSV", "LVEF"],
                   "VV_EP": ["V_TAT"],
                   "RV_volume": ["RVedv","RVesv"],
                   "RV_pressure": ["RVedp","RVpMax"],
                   "RV_dpressure": ["RVdpdtMax","RVdpdtMin"],
                   "RV_ratio": ["RVSV", "RVEF"],
                   "LA_volume": ["LAedv","LAesv","LAvMax","LAinflV"],
                   "LA_pressure": ["LApMax"],
                   "RA_volume": ["RAedv","RAesv","RAvMax","RAinflV"],
                   "RA_pressure": ["RApMax"],
                   "LA_ratio": ["LAsvA","LAsvV"],
                   "RA_ratio": ["RAsvA","RAsvV"],
                   "LV_timing": ["LVivc","LVeje","LVivr","LVfil"],
                   "RV_timing": ["RVivc","RVeje","RVivr","RVfil"],
                   }
    
    Y_original = np.loadtxt(f"{datafolder}/{output_file}", dtype=float)

    # Create Y by selecting the corresponding rows from Y_original
    Y = []
    Y_index = 0

    for i in range(len(mask)):
        if mask[i] == 1:
            Y.append(Y_original[Y_index])
            Y_index += 1
        else:
            Y.append(np.zeros(Y_original.shape[1]))

    # Convert Y to a NumPy array
    Y = np.array(Y)
    

    t = tqdm.tqdm(simulation_array, desc='Bar desc', leave=True, colour='#8B0000')
    print_message_outer = "Plotting biomarkers of cycle " 
    total_message_length_outer = len(print_message_outer) + 6

    not_plotted = []

    # Precompute label font size
    label_fontsize = plt.rcParams['axes.labelsize']
    if isinstance(label_fontsize, str):
        label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
    label_fontsize += 2

    for sim_number in t:
        if mask[sim_number] == 1:

                t.set_description(f"{print_message_outer}{sim_number}...".ljust(total_message_length_outer))

                start = int((NBEATS-1)*BCL-AVD[mask[sim_number]])
                # start = time[-1]-BCL
                end = start+BCL

                
                lv = read_csv(f"{sims_folder}/cycle_{sim_number}/cav.{chamber}.csv", delimiter=",", skipinitialspace=True,
                                header=0, comment='#')
                
                last_beat = np.intersect1d(np.where(np.array(lv['Time'])>=start)[0],
                                        np.where(np.array(lv['Time'])<=end)[0])

                time = np.array(lv['Time'][last_beat])
                volume_lv = np.array(lv['Volume'][last_beat])
                pressure_lv = np.array(lv['Pressure'][last_beat])
                dpdt_lv_ = np.diff(pressure_lv)/np.diff(time)*1000.0
                dpdt_lv = np.zeros(pressure_lv.shape,dtype=float)
                dpdt_lv[0] = dpdt_lv_[0]
                dpdt_lv[1:] = dpdt_lv_

                fig = plt.figure(figsize=(17,10))
                fig.suptitle(f"Cycle #{sim_number}", fontsize=20, weight='bold')


                pv_plt = fig.add_subplot(3,2,(1,5)) 
                p_plt = fig.add_subplot(3,2,4)
                dp_plt = fig.add_subplot(3,2,6)
                v_plt = fig.add_subplot(3,2,2)

                fig.subplots_adjust(wspace=0.3)

                pv_plt.set_xlabel('Volume (mL)')
                pv_plt.set_ylabel('Pressure (mmHg)')
                pv_plt.plot(volume_lv,pressure_lv,color='#8B0000',linewidth=2.0)

                dp_plt.set_xlabel('Time (s)')
                dp_plt.set_ylabel('Pressure rate (mmHg/ms)')
                dp_plt.plot(time,dpdt_lv,color='#8B0000',linewidth=2.0)

                p_plt.set_xlabel('Time (s)')
                p_plt.set_ylabel('Pressure (mmHg)')
                p_plt.plot(time,pressure_lv,color='#8B0000',linewidth=2.0)

                # v_plt.set_xlabel('Time (s)')
                v_plt.set_ylabel('Volume (mL)')
                v_plt.plot(time,volume_lv,color='#8B0000',linewidth=2.0)
                
                time_range = np.max(time) - np.min(time)
                volume_range = np.max(volume_lv) - np.min(volume_lv)
                pressure_range = np.max(pressure_lv) - np.min(pressure_lv)
                d_pressure_range = np.max(dpdt_lv) - np.min(dpdt_lv)

                pv_plt.set_ylim([np.min(pressure_lv)-0.1*pressure_range,
                                np.max(pressure_lv)+0.1*pressure_range])
                pv_plt.set_xlim([np.min(volume_lv)-0.1*volume_range,
                                np.max(volume_lv)+0.1*volume_range])
                
                dp_plt.set_ylim([np.min(dpdt_lv)-0.1*d_pressure_range,
                                np.max(dpdt_lv)+0.1*d_pressure_range])
                dp_plt.set_xlim([np.min(time)-0.1*time_range,
                                np.max(time)+0.1*time_range])
                
                p_plt.set_ylim([np.min(pressure_lv)-0.1*pressure_range,
                                np.max(pressure_lv)+0.1*pressure_range])
                p_plt.set_xlim([np.min(time)-0.1*time_range,
                                np.max(time)+0.1*time_range])
                
                v_plt.set_ylim([np.min(volume_lv)-0.1*volume_range,
                                np.max(volume_lv)+0.1*volume_range])
                v_plt.set_xlim([np.min(time)-0.1*time_range,
                                np.max(time)+0.1*time_range])
                v_plt.set_xticks([])
                v_plt.set_xticklabels([])

                ratio_text = []

                for label_idx, label in enumerate(ylabels):
                    if label in dict_labels[f"{chamber}_volume"]:
                        # Get the current y-axis limits
                        ymin, ymax = pv_plt.get_ylim()

                        # Draw the vertical dashed line from the bottom to the top of the plot area
                        
                        pv_plt.vlines(
                            x=Y[sim_number, label_idx], 
                            ymin=ymin, 
                            ymax=ymax, 
                            linewidth=1,        # Thinner line
                            color='k', 
                            linestyle='--'      # Dashed line
                        )

                        label_fontsize = plt.rcParams['axes.labelsize']
                        if isinstance(label_fontsize, str):
                            label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                        label_fontsize += 2

                        # Add text centered and above the line
                        pv_plt.text(
                            Y[sim_number, label_idx],    # X position (aligned with the line)
                            ymax + 0.01*pressure_range,               # Y position (at the very top of the plot area)
                            label,              # Text content
                            fontsize=label_fontsize,  # Match font size with xlabel
                            color='#8B0000',    # Dark red
                            ha='center',        # Horizontal alignment
                            va='bottom'         # Vertical alignment
                        )

                        # Get the current y-axis limits
                        xmin, xmax = v_plt.get_xlim()

                        # Draw the vertical dashed line from the bottom to the top of the plot area
                        

                        v_plt.hlines(
                            y=Y[sim_number, label_idx], 
                            xmin=xmin, 
                            xmax=xmax, 
                            linewidth=1,        # Thinner line
                            color='k', 
                            linestyle='--'      # Dashed line
                        )

                        label_fontsize = plt.rcParams['axes.labelsize']
                        if isinstance(label_fontsize, str):
                            label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                        label_fontsize += 2

                        # Add text centered and above the line
                        v_plt.text(
                            x = xmax + 0.01*volume_range,
                            y = Y[sim_number, label_idx],    # X position (aligned with the line)
                            s = label,              # Text content
                            fontsize=label_fontsize,  # Match font size with xlabel
                            color='#8B0000',    # Dark red
                            va='center',        # Central alignment
                            ha='left'         # Right alignment
                        )

                    elif label in dict_labels[f"{chamber}_pressure"]:
                        
                        # Get the current x-axis limits
                        xmin, xmax = pv_plt.get_xlim()

                        # Draw the vertical dashed line from the bottom to the top of the plot area
                        
                        pv_plt.hlines(
                            y=Y[sim_number, label_idx], 
                            xmin=xmin, 
                            xmax=xmax, 
                            linewidth=1,        # Thinner line
                            color='k', 
                            linestyle='--'      # Dashed line
                        )

                        label_fontsize = plt.rcParams['axes.labelsize']
                        if isinstance(label_fontsize, str):
                            label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                        label_fontsize += 2

                        # Add text centered and above the line
                        pv_plt.text(
                            x = xmax + 0.01*volume_range,
                            y = Y[sim_number, label_idx],    # X position (aligned with the line)
                            s = label,              # Text content
                            fontsize=label_fontsize,  # Match font size with xlabel
                            color='#8B0000',    # Dark red
                            va='center',        # Central alignment
                            ha='left'         # Right alignment
                        )



                        # Get the current x-axis limits
                        xmin, xmax = p_plt.get_xlim()

                        # Draw the vertical dashed line from the bottom to the top of the plot area
                        
                        p_plt.hlines(
                            y=Y[sim_number, label_idx], 
                            xmin=xmin, 
                            xmax=xmax, 
                            linewidth=1,        # Thinner line
                            color='k', 
                            linestyle='--'      # Dashed line
                        )

                        label_fontsize = plt.rcParams['axes.labelsize']
                        if isinstance(label_fontsize, str):
                            label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                        label_fontsize += 2

                        # Add text centered and above the line
                        p_plt.text(
                            x = xmax + 0.01*pressure_range,
                            y = Y[sim_number, label_idx],    # X position (aligned with the line)
                            s = label,              # Text content
                            fontsize=label_fontsize,  # Match font size with xlabel
                            color='#8B0000',    # Dark red
                            va='center',        # Central alignment
                            ha='left'         # Right alignment
                        )
                
                    elif label in dict_labels[f'{chamber}_dpressure']:
                        
                        # Get the current x-axis limits
                        xmin, xmax = dp_plt.get_xlim()

                        # Draw the vertical dashed line from the bottom to the top of the plot area
                        
                        dp_plt.hlines(
                            y=Y[sim_number, label_idx], 
                            xmin=xmin, 
                            xmax=xmax, 
                            linewidth=1,        # Thinner line
                            color='k', 
                            linestyle='--'      # Dashed line
                        )

                        label_fontsize = plt.rcParams['axes.labelsize']
                        if isinstance(label_fontsize, str):
                            label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                        label_fontsize += 2

                        # Add text centered and above the line
                        dp_plt.text(
                            x = xmax + 0.01*time_range,
                            y = Y[sim_number, label_idx],    # X position (aligned with the line)
                            s = label,              # Text content
                            fontsize=label_fontsize,  # Match font size with xlabel
                            color='#8B0000',    # Dark red
                            va='center',        # Central alignment
                            ha='left'         # Right alignment
                        )
                    
                    elif label in dict_labels['VV_EP']:
                        ratio_text.append(f"{label}={Y[sim_number,label_idx]} ms")
                    
                    elif label in dict_labels[f'{chamber}_ratio']:
                        ratio_text.append(f"{label}={Y[sim_number,label_idx]}")

                    elif label in dict_labels[f"{chamber}_timing"]:
                        

                        label_fontsize = plt.rcParams['axes.labelsize']
                        if isinstance(label_fontsize, str):
                            label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                        label_fontsize += 2

                        # Get the current y-axis limits
                        ymin, ymax = v_plt.get_ylim()

                        # Draw the vertical dashed line from the bottom to the top of the plot area
                        

                        v_plt.vlines(
                            x=Y[sim_number, label_idx], 
                            ymin=ymin, 
                            ymax=ymax, 
                            linewidth=1,        # Thinner line
                            color='k', 
                            linestyle='--'      # Dashed line
                        )

                        label_fontsize = plt.rcParams['axes.labelsize']
                        if isinstance(label_fontsize, str):
                            label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                        label_fontsize += 2

                        if "ivc" in label or "ivr" in label:
                            y_pos = ymax + 0.01*volume_range
                        else:
                            y_pos = ymin - 0.15*volume_range

                        # Add text centered and below the line
                        v_plt.text(
                            Y[sim_number, label_idx],    # X position (aligned with the line)
                            y_pos,               # Y position (at the very top of the plot area)
                            label,              # Text content
                            fontsize=label_fontsize,  # Match font size with xlabel
                            color='#8B0000',    # Dark red
                            ha='center',        # Horizontal alignment
                            va='bottom'         # Vertical alignment
                        )

                    else:
                        not_plotted.append(label)

                fig.text(0.5, 0.02, f"{', '.join(ratio_text)}", fontsize=14, ha='center', va='center')

                os.makedirs(f"{fig_path}/biomarkers", exist_ok=True)
                plt.savefig(f"{fig_path}/biomarkers/biomarkers_cycle_{sim_number}_{chamber}.png",dpi=300)

                plt.close()

    return(not_plotted)

def plot_biomarkers_AA(simulation_array,datafolder,mask_file,NBEATS,BCL,AVD,sims_folder,fig_path,output_file,chamber):

    print(f"Plotting biomarkers for the {chamber}...")

    
    mask = np.loadtxt(mask_file,dtype=int)

    ylabels = np.loadtxt(f"{datafolder}/ylabels.txt", dtype="str")

    dict_labels = {"LV_volume": ["LVedv","LVesv"],
                   "LV_pressure": ["LVedp","LVpMax"],
                   "LV_dpressure": ["LVdpdtMax","LVdpdtMin"],
                   "RV_volume": ["RVedv","RVesv"],
                   "RV_pressure": ["RVedp","RVpMax"],
                   "RV_dpressure": ["RVdpdtMax","RVdpdtMin"],
                   "LA_volume": ["LAedv","LAesv","LAvMax","LAinflV"],
                   "LA_pressure": ["LApMax"],
                   "RA_volume": ["RAedv","RAesv","RAvMax","RAinflV"],
                   "RA_pressure": ["RApMax"],
                   "LA_ratio": ["LAsvA","LAsvV"],
                   "RA_ratio": ["RAsvA","RAsvV"],
                   "AA_EP": ["A_TAT"]
                   }
    
    Y_original = np.loadtxt(f"{datafolder}/{output_file}", dtype=float)

    # Create Y by selecting the corresponding rows from Y_original
    Y = []
    Y_index = 0

    for i in range(len(mask)):
        if mask[i] == 1:
            Y.append(Y_original[Y_index])
            Y_index += 1
        else:
            Y.append(np.zeros(Y_original.shape[1]))

    # Convert Y to a NumPy array
    Y = np.array(Y)

    t = tqdm.tqdm(simulation_array, desc='Bar desc', leave=True, colour='#8B0000')
    print_message_outer = "Plotting biomarkers of cycle " 
    total_message_length_outer = len(print_message_outer) + 6

    not_plotted = []

    for sim_number in t:
        
        if mask[sim_number] == 1:

            t.set_description(f"{print_message_outer}{sim_number}...".ljust(total_message_length_outer))

            start = int((NBEATS-1)*BCL-AVD[mask[sim_number]])
            # start = time[-1]-BCL
            end = start+BCL

            
            lv = read_csv(f"{sims_folder}/cycle_{sim_number}/cav.{chamber}.csv", delimiter=",", skipinitialspace=True,
                            header=0, comment='#')
            
            last_beat = np.intersect1d(np.where(np.array(lv['Time'])>=start)[0],
                                    np.where(np.array(lv['Time'])<=end)[0])

            time = np.array(lv['Time'][last_beat])
            volume_lv = np.array(lv['Volume'][last_beat])
            pressure_lv = np.array(lv['Pressure'][last_beat])
            
            fig = plt.figure(figsize=(17,10))
            fig.suptitle(f"Cycle #{sim_number}", fontsize=20, weight='bold')


            pv_plt = fig.add_subplot(2,2,(1,3)) 
            p_plt = fig.add_subplot(2,2,2)
            v_plt = fig.add_subplot(2,2,4)

            fig.subplots_adjust(wspace=0.3)

            pv_plt.set_xlabel('Volume (mL)')
            pv_plt.set_ylabel('Pressure (mmHg)')
            pv_plt.plot(volume_lv,pressure_lv,color='#8B0000',linewidth=2.0)

            p_plt.set_xlabel('Time (s)')
            p_plt.set_ylabel('Pressure (mmHg)')
            p_plt.plot(time,pressure_lv,color='#8B0000',linewidth=2.0)

            v_plt.set_xlabel('Time (s)')
            v_plt.set_ylabel('Volume (mL)')
            v_plt.plot(time,volume_lv,color='#8B0000',linewidth=2.0)
            
            time_range = np.max(time) - np.min(time)
            volume_range = np.max(volume_lv) - np.min(volume_lv)
            pressure_range = np.max(pressure_lv) - np.min(pressure_lv)

            pv_plt.set_ylim([np.min(pressure_lv)-0.1*pressure_range,
                             np.max(pressure_lv)+0.1*pressure_range])
            pv_plt.set_xlim([np.min(volume_lv)-0.1*volume_range,
                             np.max(volume_lv)+0.1*volume_range])
            
            p_plt.set_ylim([np.min(pressure_lv)-0.1*pressure_range,
                             np.max(pressure_lv)+0.1*pressure_range])
            p_plt.set_xlim([np.min(time)-0.1*time_range,
                             np.max(time)+0.1*time_range])
            
            v_plt.set_ylim([np.min(volume_lv)-0.1*volume_range,
                             np.max(volume_lv)+0.1*volume_range])
            v_plt.set_xlim([np.min(time)-0.1*time_range,
                             np.max(time)+0.1*time_range])

            ratio_text = []

            for label_idx, label in enumerate(ylabels):
                if label in dict_labels[f"{chamber}_volume"]:
                    
                    # Get the current y-axis limits
                    ymin, ymax = pv_plt.get_ylim()

                    # Draw the vertical dashed line from the bottom to the top of the plot area
                    
                    pv_plt.vlines(
                        x=Y[sim_number, label_idx], 
                        ymin=ymin, 
                        ymax=ymax, 
                        linewidth=1,        # Thinner line
                        color='k', 
                        linestyle='--'      # Dashed line
                    )

                    label_fontsize = plt.rcParams['axes.labelsize']
                    if isinstance(label_fontsize, str):
                        label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                    label_fontsize += 2

                    # Add text centered and above the line
                    pv_plt.text(
                        Y[sim_number, label_idx],    # X position (aligned with the line)
                        ymax + 0.01*pressure_range,               # Y position (at the very top of the plot area)
                        label,              # Text content
                        fontsize=label_fontsize,  # Match font size with xlabel
                        color='#8B0000',    # Dark red
                        ha='center',        # Horizontal alignment
                        va='bottom'         # Vertical alignment
                    )

                    # Get the current y-axis limits
                    xmin, xmax = v_plt.get_xlim()

                    # Draw the vertical dashed line from the bottom to the top of the plot area
                    

                    v_plt.hlines(
                        y=Y[sim_number, label_idx], 
                        xmin=xmin, 
                        xmax=xmax, 
                        linewidth=1,        # Thinner line
                        color='k', 
                        linestyle='--'      # Dashed line
                    )

                    label_fontsize = plt.rcParams['axes.labelsize']
                    if isinstance(label_fontsize, str):
                        label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                    label_fontsize += 2

                    # Add text centered and above the line
                    v_plt.text(
                        x = xmax + 0.01*volume_range,
                        y = Y[sim_number, label_idx],    # X position (aligned with the line)
                        s = label,              # Text content
                        fontsize=label_fontsize,  # Match font size with xlabel
                        color='#8B0000',    # Dark red
                        va='center',        # Central alignment
                        ha='left'         # Right alignment
                    )

                elif label in dict_labels[f"{chamber}_pressure"]:
                    
                    # Get the current x-axis limits
                    xmin, xmax = pv_plt.get_xlim()

                    # Draw the vertical dashed line from the bottom to the top of the plot area
                    
                    pv_plt.hlines(
                        y=Y[sim_number, label_idx], 
                        xmin=xmin, 
                        xmax=xmax, 
                        linewidth=1,        # Thinner line
                        color='k', 
                        linestyle='--'      # Dashed line
                    )

                    label_fontsize = plt.rcParams['axes.labelsize']
                    if isinstance(label_fontsize, str):
                        label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                    label_fontsize += 2

                    # Add text centered and above the line
                    pv_plt.text(
                        x = xmax + 0.01*volume_range,
                        y = Y[sim_number, label_idx],    # X position (aligned with the line)
                        s = label,              # Text content
                        fontsize=label_fontsize,  # Match font size with xlabel
                        color='#8B0000',    # Dark red
                        va='center',        # Central alignment
                        ha='left'         # Right alignment
                    )



                    # Get the current x-axis limits
                    xmin, xmax = p_plt.get_xlim()

                    # Draw the vertical dashed line from the bottom to the top of the plot area
                    
                    p_plt.hlines(
                        y=Y[sim_number, label_idx], 
                        xmin=xmin, 
                        xmax=xmax, 
                        linewidth=1,        # Thinner line
                        color='k', 
                        linestyle='--'      # Dashed line
                    )

                    label_fontsize = plt.rcParams['axes.labelsize']
                    if isinstance(label_fontsize, str):
                        label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                    label_fontsize += 2

                    # Add text centered and above the line
                    p_plt.text(
                        x = xmax + 0.01*pressure_range,
                        y = Y[sim_number, label_idx],    # X position (aligned with the line)
                        s = label,              # Text content
                        fontsize=label_fontsize,  # Match font size with xlabel
                        color='#8B0000',    # Dark red
                        va='center',        # Central alignment
                        ha='left'         # Right alignment
                    )

                elif label in dict_labels[f'{chamber}_ratio']:
                    ratio_text.append(f"{label}={Y[sim_number,label_idx]}")
                
                elif label in dict_labels['AA_EP']:
                    ratio_text.append(f"{label}={Y[sim_number,label_idx]} ms")

                else:
                    not_plotted.append(label)

            fig.text(0.5, 0.02, f"{', '.join(ratio_text)}", fontsize=14, ha='center', va='center')

            os.makedirs(f"{fig_path}/biomarkers", exist_ok=True)
            plt.savefig(f"{fig_path}/biomarkers/biomarkers_cycle_{sim_number}_{chamber}.png",dpi=300)

            plt.close()
        
    return(not_plotted)

def plot_biomarkers_arteries(simulation_array,datafolder,mask_file,NBEATS,BCL,AVD,sims_folder,fig_path,output_file):

    print(f"Plotting biomarkers for the arteries...")

    
    mask = np.loadtxt(mask_file,dtype=int)

    ylabels = np.loadtxt(f"{datafolder}/ylabels.txt", dtype="str")

    dict_labels = {"ao_pressure": ["diastAP","systAP","mAP"],
                   "ao_ratio": ["pulseAP"],
                   "pa_pressure": ["diastPAP","systPAP","mPAP"],
                   "pa_ratio": ["pulsePAP"]
                   }
    
    Y_original = np.loadtxt(f"{datafolder}/{output_file}", dtype=float)

    # Create Y by selecting the corresponding rows from Y_original
    Y = []
    Y_index = 0

    for i in range(len(mask)):
        if mask[i] == 1:
            Y.append(Y_original[Y_index])
            Y_index += 1
        else:
            Y.append(np.zeros(Y_original.shape[1]))

    # Convert Y to a NumPy array
    Y = np.array(Y)

    t = tqdm.tqdm(simulation_array, desc='Bar desc', leave=True, colour='#8B0000')
    print_message_outer = "Plotting biomarkers of cycle " 
    total_message_length_outer = len(print_message_outer) + 6

    not_plotted = []

    for sim_number in t:
        
        if mask[sim_number] == 1:

            t.set_description(f"{print_message_outer}{sim_number}...".ljust(total_message_length_outer))

            start = int((NBEATS-1)*BCL-AVD[mask[sim_number]])
            # start = time[-1]-BCL
            end = start+BCL

            
            ao = read_csv(f"{sims_folder}/cycle_{sim_number}/tube.AO.csv", delimiter=",", skipinitialspace=True,
                            header=0, comment='#')
            pa = read_csv(f"{sims_folder}/cycle_{sim_number}/tube.AP.csv", delimiter=",", skipinitialspace=True,
                            header=0, comment='#')
            
            last_beat = np.intersect1d(np.where(np.array(ao['Time'])>=start)[0],
                                    np.where(np.array(ao['Time'])<=end))
            
            time = np.array(ao['Time'][last_beat])

            pressure_ao = np.array(ao['Pressure'][last_beat])
            pressure_pa = np.array(pa['Pressure'][last_beat])
            
            fig = plt.figure(figsize=(17,10))
            fig.suptitle(f"Cycle #{sim_number}", fontsize=20, weight='bold')


            ap_plt = fig.add_subplot(2,1,1)
            pap_plt = fig.add_subplot(2,1,2)

            ap_plt.set_xlabel('Time (s)')
            ap_plt.set_ylabel('Pressure (mmHg)')
            ap_plt.plot(time,pressure_ao,color='#8B0000',linewidth=2.0)

            pap_plt.set_xlabel('Time (s)')
            pap_plt.set_ylabel('Pressure (mmHg)')
            pap_plt.plot(time,pressure_pa,color='#8B0000',linewidth=2.0)

            
            time_range = np.max(time) - np.min(time)
            ao_pressure_range = np.max(pressure_ao) - np.min(pressure_ao)
            pa_pressure_range = np.max(pressure_pa) - np.min(pressure_pa)


            ap_plt.set_ylim([np.min(pressure_ao)-0.1*ao_pressure_range,
                             np.max(pressure_ao)+0.1*ao_pressure_range])
            ap_plt.set_xlim([np.min(time)-0.1*time_range,
                             np.max(time)+0.1*time_range])
            
            pap_plt.set_ylim([np.min(pressure_pa)-0.1*pa_pressure_range,
                             np.max(pressure_pa)+0.1*pa_pressure_range])
            pap_plt.set_xlim([np.min(time)-0.1*time_range,
                             np.max(time)+0.1*time_range])

            ratio_text = []

            for label_idx, label in enumerate(ylabels):

                if label in dict_labels[f"ao_pressure"]:
                    
                    # Get the current x-axis limits
                    xmin, xmax = ap_plt.get_xlim()

                    label_fontsize = plt.rcParams['axes.labelsize']
                    if isinstance(label_fontsize, str):
                        label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                    label_fontsize += 2

                    # Get the current x-axis limits
                    xmin, xmax = ap_plt.get_xlim()

                    # Draw the vertical dashed line from the bottom to the top of the plot area
                    
                    ap_plt.hlines(
                        y=Y[sim_number, label_idx], 
                        xmin=xmin, 
                        xmax=xmax, 
                        linewidth=1,        # Thinner line
                        color='k', 
                        linestyle='--'      # Dashed line
                    )

                    label_fontsize = plt.rcParams['axes.labelsize']
                    if isinstance(label_fontsize, str):
                        label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                    label_fontsize += 2

                    # Add text centered and above the line
                    ap_plt.text(
                        x = xmax + 0.01*ao_pressure_range,
                        y = Y[sim_number, label_idx],    # X position (aligned with the line)
                        s = label,              # Text content
                        fontsize=label_fontsize,  # Match font size with xlabel
                        color='#8B0000',    # Dark red
                        va='center',        # Central alignment
                        ha='left'         # Right alignment
                    )

                elif label in dict_labels[f'ao_ratio']:
                    ratio_text.append(f"Ao {label}={Y[sim_number,label_idx]}")
                
                elif label in dict_labels[f"pa_pressure"]:
                    
                    # Get the current x-axis limits
                    xmin, xmax = pap_plt.get_xlim()

                    label_fontsize = plt.rcParams['axes.labelsize']
                    if isinstance(label_fontsize, str):
                        label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                    label_fontsize += 2

                    # Get the current x-axis limits
                    xmin, xmax = pap_plt.get_xlim()

                    # Draw the vertical dashed line from the bottom to the top of the plot area
                    
                    pap_plt.hlines(
                        y=Y[sim_number, label_idx], 
                        xmin=xmin, 
                        xmax=xmax, 
                        linewidth=1,        # Thinner line
                        color='k', 
                        linestyle='--'      # Dashed line
                    )

                    label_fontsize = plt.rcParams['axes.labelsize']
                    if isinstance(label_fontsize, str):
                        label_fontsize = 12 if label_fontsize == "medium" else 10  # Default fallback for common values
                    label_fontsize += 2

                    # Add text centered and above the line
                    pap_plt.text(
                        x = xmax + 0.01*pa_pressure_range,
                        y = Y[sim_number, label_idx],    # X position (aligned with the line)
                        s = label,              # Text content
                        fontsize=label_fontsize,  # Match font size with xlabel
                        color='#8B0000',    # Dark red
                        va='center',        # Central alignment
                        ha='left'         # Right alignment
                    )

                elif label in dict_labels[f'pa_ratio']:
                    ratio_text.append(f"PA {label}={Y[sim_number,label_idx]}")
                
                else:
                    not_plotted.append(label)

            fig.text(0.5, 0.02, f"{', '.join(ratio_text)}", fontsize=14, ha='center', va='center')

            os.makedirs(f"{fig_path}/biomarkers", exist_ok=True)
            plt.savefig(f"{fig_path}/biomarkers/biomarkers_cycle_{sim_number}_arteries.png",dpi=300)

            plt.close()
        
    return(not_plotted)


def print_PV_loops_all_cycles(path_to_simulation, BCL, case_number):
    
    chambers = ['LV', 'RV', 'LA', 'RA']
    colours = ['red', 'blue', '#F6BE00', 'green']
    
    ax = plt.figure(figsize=(10,10), constrained_layout=True).subplots(2, 2)
    ax = ax.flatten()
    n_cycles = 5

    for j, chamber_name in enumerate(chambers):
        
        for n in range(n_cycles):

            chamber_structure = read_csv(os.path.join(path_to_simulation,'cav.'+chamber_name+'.csv'), delimiter=",",
                          skipinitialspace=True, header=0, comment='#')
            time = np.array(chamber_structure['Time'])

    #         start = time[-1]-BCL
            start = time[0] + n*BCL
            end = start + BCL

            plot_time = np.where((time>=start) & (time<end))[0]

            volume = np.array(chamber_structure['Volume'][plot_time])
            pressure = np.array(chamber_structure['Pressure'][plot_time])
            ax[j].plot(volume,pressure,color=colours[j],linewidth=3.0, alpha = 0.1+n*((1-0.1)/n_cycles))
            ax[j].set_xlabel(chamber_name+' volume [mL]')
            ax[j].set_ylabel(chamber_name+' pressure [mmHg]')
        EF = round(100*(np.max(volume)-np.min(volume))/np.max(volume),2)
        ax[j].text(0.95, 0.95, 'EF: ' + str(EF) + "%", horizontalalignment='right', verticalalignment='top',
                transform=ax[j].transAxes)
    plt.suptitle(f"Simulation #{case_number}", fontsize=20, weight='bold')
    os.makedirs(f"{path_to_simulation}/../../figures/PV_loops/", exist_ok=True)
    plt.savefig(f"{path_to_simulation}/../../figures/PV_loops/{case_number}_pv_loops_all_cycles.png",dpi=300)
    plt.close('all')
    

def plot_pvloops_all_sim_range(datafolder,
                      output_folder,
                     BCL,
                     simulation_array,
                     basename="cycle_",
                     figname=None,
                     mask_file=None,
                     default_sim = False):

    print('Plotting PV loops for successful simulations...')

    chambers = ['LV','RV','LA','RA']

    if mask_file is None:
        mask_file = datafolder+"/output_mask.txt"
    
    mask = np.loadtxt(mask_file,dtype=int)
    
    ax = plt.figure(figsize=(10,10), constrained_layout=True).subplots(2, 2)
    ax = ax.flatten()

    for i in simulation_array:

        if mask[i] == 1:
        
            for j,c in enumerate(chambers):

                if default_sim:
                    ch = read_csv(output_folder+'/'+basename+'default/cav.'+c+'.csv', delimiter=",", skipinitialspace=True,
                                        header=0, comment='#')
                else:
                    ch = read_csv(f"{output_folder}/{basename}{i}/cav.{c}.csv", delimiter=",", skipinitialspace=True,
                                        header=0, comment='#')      
                time = np.array(ch['Time'])

                start = time[-1]-BCL

                plot_time = np.where(time>=start)[0]

                volume = np.array(ch['Volume'][plot_time])
                pressure = np.array(ch['Pressure'][plot_time])
                t = np.array(ch['Time'][plot_time])      

                ax[j].plot(volume,pressure,color='#3489eb')
                ax[j].set_xlabel(c+' volume [mL]')
                ax[j].set_ylabel(c+' pressure [mmHg]')

    if figname is not None:
        plt.savefig(figname,dpi=300)
    else:
        plt.show()



def main(args):

    basefolder         = args.basefolder
    simulations_folder = f"{basefolder}/simulations"
    output_folder      = f"{basefolder}/output"
    figures_path       = f"{basefolder}/figures"
    n_beat             = args.n_beat
    first_simulation   = args.first_simulation
    last_simulation    = args.last_simulation
    default            = args.default
    output_file        = args.output_file
    simulation_array   = args.simulation_array

    with open(f"{basefolder}/json_files/clinical_data.json", "r") as clinical_data:
        clinical_json = json.load(clinical_data)
    
    BCL = clinical_json["general"]["BCL"]

    X   = np.loadtxt(f"{basefolder}/data/X.txt")
    with open(f'{basefolder}/data/xlabels.txt', 'r') as file:
            xlabels = file.read().splitlines()

    AVD_initial = X[:, xlabels.index('AV_delay')]


    if first_simulation is not None:
        simulation_array = range(first_simulation, last_simulation+1)


    AVD = AVD_initial[simulation_array]

    not_plotted_LV = plot_biomarkers_VV(simulation_array = simulation_array,
                    datafolder = f"{basefolder}/data",
                    mask_file = f"{output_folder}/output_mask_beat_{n_beat}.txt",
                    NBEATS = n_beat,
                    BCL = BCL,
                    AVD = AVD,
                    sims_folder = simulations_folder,
                    fig_path = figures_path,
                    output_file=output_file,
                    chamber="LV")
    not_plotted_RV = plot_biomarkers_VV(simulation_array=simulation_array,
                    datafolder = f"{basefolder}/data",
                    mask_file = f"{output_folder}/output_mask_beat_{n_beat}.txt",
                    NBEATS = n_beat,
                    BCL = BCL,
                    AVD = AVD,
                    sims_folder = simulations_folder,
                    fig_path = figures_path,
                    output_file=output_file,
                    chamber="RV")
    
    not_plotted_LA = plot_biomarkers_AA(simulation_array=simulation_array,
                    datafolder = f"{basefolder}/data",
                    mask_file = f"{output_folder}/output_mask_beat_{n_beat}.txt",
                    NBEATS = n_beat,
                    BCL = BCL,
                    AVD = AVD,
                    sims_folder = simulations_folder,
                    fig_path = figures_path,
                    output_file=output_file,
                    chamber="LA")
    not_plotted_RA = plot_biomarkers_AA(simulation_array=simulation_array,
                    datafolder = f"{basefolder}/data",
                    mask_file = f"{output_folder}/output_mask_beat_{n_beat}.txt",
                    NBEATS = n_beat,
                    BCL = BCL,
                    AVD = AVD,
                    sims_folder = simulations_folder,
                    fig_path = figures_path,
                    output_file=output_file,
                    chamber="RA")
    
    not_plotted_arteries = plot_biomarkers_arteries(simulation_array=simulation_array,
                    datafolder = f"{basefolder}/data",
                    mask_file = f"{output_folder}/output_mask_beat_{n_beat}.txt",
                    NBEATS = n_beat,
                    BCL = BCL,
                    AVD = AVD,
                    sims_folder = simulations_folder,
                    fig_path = figures_path,
                    output_file=output_file)


    common_biomarkers = set(not_plotted_LV) & set(not_plotted_RV) &  set(not_plotted_LA) & set(not_plotted_RA) & set(not_plotted_arteries)

    if common_biomarkers:
        print(f"Biomarkers not plotted: {common_biomarkers}")
      
    plot_pvloops_all_sim_range(datafolder    = output_folder,
                                        output_folder = simulations_folder,
                                        BCL           = BCL,
                                        basename      = "cycle_",
                                        mask_file     = f"{output_folder}/output_mask_beat_{n_beat}.txt",
                                        simulation_array=simulation_array,
                                        figname       = f"{figures_path}/all_pv_loops_beat_{n_beat}.png")


    if not default:


        output_mask = np.loadtxt(f"{output_folder}/output_mask_beat_{n_beat}.txt")

        
        t = tqdm.tqdm(simulation_array, desc='Bar desc', leave=True,colour='#80EF80')

        for sim_number in t:
            if output_mask[sim_number] == 1:
                t.set_description(f"Plotting PV loops of simulation #{sim_number}...")
                path2simulation = f"{simulations_folder}/cycle_{sim_number}"
                
                print_PV_loops_all_cycles(path_to_simulation = path2simulation,
                                          BCL = BCL, case_number=sim_number)
    else:
        print_PV_loops_all_cycles(path_to_simulation = f"{simulations_folder}/cycle_default",BCL = BCL, case_number='default')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    
    parser.add_argument('--basefolder', type=str, required=True,
                        default=os.path.join(os.environ.get("DATA_ROOT", ""), "simulations"),
                        help='Path to the folder where the simulations, data, and figure folders are.')
    parser.add_argument('--n_beat', type=int, required=False, help="Heartbeat number to compute the output.", default=5)
    parser.add_argument('--first_simulation', type=int, help="First simulation index.", default=None)
    parser.add_argument('--last_simulation', type=int, help="Last simulation index.", default=None)
    parser.add_argument('--simulation_array', type=int, nargs='*', help="Array of simulation indices.")
    parser.add_argument('--default', action='store_true')
    parser.add_argument('--output_file', default="Y.txt")

    args = parser.parse_args()

    main(args)
