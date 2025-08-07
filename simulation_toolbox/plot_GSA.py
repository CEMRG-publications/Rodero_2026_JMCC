import os
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re  # Import regex module for sanitizing filenames
from PIL import Image
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
import matplotlib.cm as cm
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.collections import LineCollection
import json
from matplotlib.lines import Line2D

def format_title_with_bold_latex(title):
    parts = title.split('$$')
    for i in range(len(parts)):
        if i % 2 == 0:
            # Non-math part: wrap in \textbf{}
            parts[i] = r'\textbf{' + parts[i] + '}'
        else:
            # Math part: wrap in \boldsymbol{}
            parts[i] = r'\boldsymbol{' + parts[i] + '}'
    return ''.join(parts)

def read_labels(file_path):
    return np.loadtxt(file_path, dtype=str)

def sanitize_filename(filename):
    """
    Sanitize the filename by removing or replacing invalid characters.
    Replace multiple consecutive underscores with a single underscore.
    """
    # Replace LaTeX special characters with underscores or remove them
    sanitized = re.sub(r'[^\w\-_\. ]', '_', filename)
    # Replace multiple underscores with a single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized

def plot_GSA_radar_chart(scenarios, xlabels, ylabels, savepath, fontsize, figname_preffix, legend=[], colors=[], threshold=0):
    
    if os.path.exists(f"{scenarios[0]}/data/features_idx_list_gsa.txt"):
        # Read the features index list from the file
        print(f"Reading features index list from {scenarios[0]}/data/features_idx_list_gsa.txt")
        features_idx_list = np.loadtxt(f"{scenarios[0]}/data/features_idx_list_gsa.txt", dtype=int)
    else:
        # If the file does not exist, create a default list
        print(f"File {scenarios[0]}/data/features_idx_list_gsa.txt not found. Using default index list.")
        features_idx_list = list(range(len(ylabels)))

    for loop_idx, output_idx in enumerate(features_idx_list):
        ylabel_name = ylabels[output_idx]

        # Sanitize the filename to ensure it is valid
        sanitized_ylabel_name = sanitize_filename(ylabel_name)
        figname = f"{figname_preffix}_{sanitized_ylabel_name}"

        theta = xlabels

        r = np.zeros((len(scenarios), len(theta)))

        for i, scenario in enumerate(scenarios):
            # Read Si_total.csv and convert to numpy array of floats
            with open(f"{scenario}/output/Si_total.csv", 'r') as f:
                csv_reader = csv.reader(f)
                ST_all = np.array([list(map(float, row)) for row in csv_reader])

            order_effects = ST_all[:, loop_idx]
            order_effects = order_effects / np.sum(order_effects)

            r[i] = order_effects

        # Filter theta and r based on the threshold
        filtered_indices = [i for i in range(len(theta)) if np.any(r[:, i] >= threshold)]
        filtered_theta = [theta[i] for i in filtered_indices]
        filtered_r = r[:, filtered_indices]

        # Recompute theta radians to be equispaced
        theta_radians = np.linspace(0, 2 * np.pi, len(filtered_theta), endpoint=False).tolist()
        theta_radians += theta_radians[:1]

        filtered_r = np.concatenate((filtered_r, filtered_r[:, :1]), axis=1)

        if len(colors) == 0:
            colors = ['#e662ad', '#f7c26f', '#6ecb74', '#173263']

        # Create polar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        if len(legend) == 0:
            legend = [f"Heart #{i+1}" for i in range(len(scenarios))]

        for i, scenario in enumerate(scenarios):
            ax.plot(theta_radians, filtered_r[i], marker='.', label=legend[i], color=colors[i])

        # Set theta ticks and labels
        ax.set_xticks(theta_radians[:-1])
        ax.set_xticklabels(filtered_theta, fontsize=fontsize)

        # Adjust alignment of x-tick labels
        for label, angle in zip(ax.get_xticklabels(), theta_radians[:-1]):
            if angle < np.pi/2.0 or angle > 3*np.pi/2.0:  # Far-right label
                label.set_horizontalalignment('left')
            else:  
                label.set_horizontalalignment('right')

        ax.xaxis.set_tick_params(which="major", pad=8)

        # Set radial axis ticks and labels, excluding 1
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])  # Exclude 0 and 1
        ax.tick_params(axis='y', labelsize=fontsize - 2)  # Reduce radial label font size

        ax.set_rmax(1)

        title = f"{ylabels[output_idx]}"
        if title.startswith("$"):
            title = f"$\mathbf{{{title[1:-1]}}}$"

        plt.title(title, fontsize=fontsize+5, fontweight="bold", pad=30)
        plt.legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

        plt.tight_layout()

        os.makedirs(savepath, exist_ok=True)
        save_file_path = f"{savepath}/{figname}.png"
        plt.savefig(save_file_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to: {save_file_path}")

        # Close the figure to free memory
        plt.close(fig)

def plot_GSA_radar_chart_feature(scenarios, xlabels_plain_all, xlabels_dict_all, ylabels_latex_all, ylabels_raw_all, savepath, fontsize, figname_preffix, legend=[], 
                                 colors=[], threshold=0, feature_idx=0, loop_idx=0):
    

    output_idx = feature_idx
    # ylabel_name = ylabels[output_idx]

    # Sanitize the filename to ensure it is valid
    # sanitized_ylabel_name = sanitize_filename(ylabel_name)
    # figname = f"{figname_preffix}_{sanitized_ylabel_name}"

    theta = [xlabels_dict_all[i]['latex'] for i in xlabels_plain_all]


    r = np.zeros((len(scenarios), len(theta)))

    for i, scenario in enumerate(scenarios):
        # Read Si_total.csv and convert to numpy array of floats
        with open(f"{scenario}/output/Si_total.csv", 'r') as f:
            csv_reader = csv.reader(f)
            ST_all = np.array([list(map(float, row)) for row in csv_reader])

        order_effects = ST_all[:, loop_idx]
        order_effects = order_effects / np.sum(order_effects)

        r[i] = order_effects

    # Filter theta and r based on the threshold
    filtered_indices = [i for i in range(len(theta)) if np.any(r[:, i] >= threshold)]
    filtered_theta = [theta[i] for i in filtered_indices]
    filtered_r = r[:, filtered_indices]

    # Recompute theta radians to be equispaced
    theta_radians = np.linspace(0, 2 * np.pi, len(filtered_theta), endpoint=False).tolist()
    theta_radians += theta_radians[:1]

    filtered_r = np.concatenate((filtered_r, filtered_r[:, :1]), axis=1)

    if len(colors) == 0:
        colors = ['#e662ad', '#f7c26f', '#6ecb74', '#173263']

    # Create polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    if len(legend) == 0:
        legend = [f"Heart #{i+1}" for i in range(len(scenarios))]

    for i, scenario in enumerate(scenarios):
        ax.plot(theta_radians, filtered_r[i], marker='.', label=legend[i], color=colors[i])

    # Set theta ticks and labels
    ax.set_xticks(theta_radians[:-1])
    ax.set_xticklabels(filtered_theta, fontsize=fontsize)

    # Adjust alignment of x-tick labels
    for label, angle in zip(ax.get_xticklabels(), theta_radians[:-1]):
        if angle < np.pi/2.0 or angle > 3*np.pi/2.0:  # Far-right label
            label.set_horizontalalignment('left')
        else:  
            label.set_horizontalalignment('right')

    ax.xaxis.set_tick_params(which="major", pad=8)

    # Set radial axis ticks and labels, excluding 1
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])  # Exclude 0 and 1
    ax.tick_params(axis='y', labelsize=fontsize - 2)  # Reduce radial label font size

    ax.set_rmax(1)

    title = f"{ylabels_latex_all[output_idx]}"
    if title.startswith("$"):
        title = f"$\mathbf{{{title[1:-1]}}}$"

    plt.title(title, fontsize=fontsize+5, fontweight="bold", pad=30)
    plt.legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))

    plt.tight_layout()

    os.makedirs(savepath, exist_ok=True)
    save_file_path = f"{savepath}/{ylabels_raw_all[output_idx]}.png"
    plt.savefig(save_file_path, bbox_inches="tight", dpi=300)
    print(f"Plot saved to: {save_file_path}")

    # Close the figure to free memory
    plt.close(fig)


def plot_rank_GSA_free_th_color(datapath,
                                loadpath,
                                xlabels,
                                rank_file=None,
                                criterion="STi",
                                mode="max",
                                figname="",
                                normalise=False,
                                acc_var_th=0.9,
                                annotate=False,
                                fontsize=14,
                                separate_colors=False,
                                color_important=None,
                                color_all="#ff8000",
                                title=None,
                                top_n=100):
    """
    Plots the parameter ranking for a GSA.

    Args:
        - datapath: folder with data (xlabels.txt, etc...)
        - loadpath: path where you saved your parameter ranking 
        - rank_file: if you want to provide a different parameter ranking file 
                     that is not in the loadpath folder
        - criterion: STi or Si e.g. total or first order effects to use for ranking
        - mode: max or mean to rank the parameters
        - figname: name of output figure 
        - normalise: if you want to normalise so that the values all sum up to 1 
        - th: threshold to determine which parameters are important and which ones aren't
        - annotate: write numbers on top of each bar
        - figsize: size of output figure (width, height). If None, width is adjusted dynamically.
        - fontsize: size of figure font
        - separate_colors: if you want a different colour for important and unimportant parameters
        - color_important: what colour you want the important parameter bars to be
        - top_n: number of top-ranked parameters to show
    """

    color = [color_all]

    # Read xlabels
    # index_i = read_labels(datapath + "/xlabels_plot.txt")
    index_i = xlabels

    # Dynamically adjust the width of the plot based on the number of parameters
    width = max(15, len(index_i)-15) 
    # print(f"Width of the plot: {width}")
    figsize = (width, 7)

    x = np.arange(len(index_i))
    barWidth = 0.4

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        

    if rank_file is None:
        rank_file = loadpath + "/Rank_" + criterion + "_" + mode + ".txt"

    f = open(rank_file, "r")
    lines = f.readlines()

    r_dct = {}
    for line in lines:
        line_split = re.split(r'\t+', line)
        r_dct[line_split[0]] = float(line_split[1])

    bars = []
    
    for l in index_i:
        bars.append(r_dct[l])
    idx_sorted = np.argsort(np.array(bars))

    bars_sorted = [100*bars[idx] for idx in idx_sorted]
    bars_sorted = bars_sorted[::-1]
    
    index_i_sorted = [index_i[idx] for idx in idx_sorted]
    index_i_sorted = index_i_sorted[::-1]

    # Limit to top_n parameters
    bars_sorted = bars_sorted[:top_n]
    index_i_sorted = index_i_sorted[:top_n]
    
    # Update x-axis and figure width based on top_n
    x = np.arange(len(index_i_sorted))
    width = max(15, len(index_i_sorted)-15) 

    longest_label = max(len(label) for label in index_i_sorted)
    height = max(7, longest_label/10 + 3)  # Adjust height based on label length

    figsize = (width, height)
    fig.set_size_inches(figsize)
    
    r = [xx + barWidth for xx in x]
    
    ax.set_xlim(min(r) - barWidth, max(r) + barWidth)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    bars_sorted_norm = list(np.array(bars_sorted) / sum(bars_sorted))

    bars_sorted_sum = []
    for i in range(len(bars_sorted)):
        bars_sorted_sum.append(sum(bars_sorted_norm[0:i+1]))

    if normalise:
        barplot = bars_sorted_norm
    else:
        barplot = bars_sorted

    plt.xticks(x + barWidth, index_i_sorted, rotation=45, fontsize=fontsize, ha='right')
    ax.tick_params(axis='both', labelsize=fontsize)

    cutoff_param = np.where(np.array(bars_sorted_sum) > acc_var_th)[0][0] if len(np.where(np.array(bars_sorted_sum) > acc_var_th)[0]) > 0 else len(bars_sorted_sum) - 1
    if color_important is None:
        color_important = "#c08978"
    color_unimportant = "lightgray"
    colors = [color_important] * (cutoff_param + 1) + [color_unimportant] * (len(bars_sorted_sum) - cutoff_param - 1)

    if separate_colors:
        bars = ax.bar(r, barplot, width=barWidth, edgecolor='white', color=colors)
    else:
        bars = ax.bar(r, barplot, color=color, width=barWidth, edgecolor='white')

    if criterion == 'Si_total':
        ax.set_ylabel('Maximum variance explained (%)', fontsize=fontsize)
    else:
        ax.set_ylabel('First-order effects', fontsize=fontsize)
    
    # if th > 0:
    #     ax.plot([-2 * barWidth, len(index_i) + 2 * barWidth], [th, th], color='black', linestyle='--')
    
    # plt.legend()

    meshname = loadpath.split("/")[-6]

    if title is None:
        title = f"Parameter ranking for mesh #{meshname} (Top {top_n})"

    ax.set_title(title, fontsize=fontsize + 5, fontweight="bold")

    # y_max = np.ceil(np.max(np.array(barplot)) * 10) / 10

    # cutoff = r[cutoff_param] + 2 * barWidth
    # ax.plot([cutoff, cutoff], [0, y_max], color='black', linestyle='--')

    if annotate:
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            ax.text(bar.get_x(), yval + .005, str(round(bars_sorted_sum[i] * 100)) + '%', fontsize=fontsize)

    plt.savefig(figname)
    print(f"Figure saved to: {figname}")

    # Close the figure to free memory
    plt.close(fig)

def gsa_parameters_ranking_S_free_pathfile(loadpath,
                                           loadpath_sobol,
                                           xlabels,
                                           ylabels_raw_all,
                                           features,
                                           gsa_mode="STi",
                                           mode="max",
                                           threshold_cutoff=0.9,
                                           output_file=None,
                                           features_file=None,
                                           important_params_idx_file=None,
                                           ):
    """
    Ranks parameters using sensitivity indices.

    Args:
        - loadpath: datafolder containing data used for GPE training
        - loadpath_sobol: folder containing GPEs and GSA results
        - gsa_mode: output GSA file to read in 
        - mode: how to rank the parameters [max,mean,sum]
        - output_file: output file to save ranking
        - features_file: features file containing which features to consider
    """
    
    # Read xlabels (parameter names)
    # ylabels_all = read_labels(loadpath + "ylabels.txt")  # Read ylabels for naming files

    # if features_file is None:
    #     features_file = loadpath + "features_idx_list_gsa.txt"

    # if not os.path.exists(features_file):
    #     raise Exception('Cannot find ' + features_file)

    # features = np.loadtxt(features_file, dtype=int)
    ylabels = [ylabels_raw_all[i] for i in features]

    # Load the sensitivity index data (S) from the Sobol analysis file
    with open(f"{loadpath_sobol}/{gsa_mode}.csv", 'r') as f:
        csv_reader = csv.reader(f)
        S = np.array([list(map(float, row)) for row in csv_reader])

    # Rank parameters based on the global effect (max across all labels)
    S_total = np.zeros((len(xlabels), 1), dtype=float)
    print('Ranking parameters according to their ' + mode + ' effect...')

    if mode == "mean":
        S_total = np.mean(S, axis=1)
    elif mode == "sum":
        S_total = np.sum(S, axis=1)
    elif mode == "max":
        S_total = np.max(S, axis=1)
    else:
        print("mode not recognised: please choose between mean, max and sum")

    ranked = np.argsort(S_total)
    ranked = ranked[::-1]  # Reverse to get highest to lowest
    ranked_S = S_total[ranked]

    # Output the global ranking to a file
    if output_file is None:
        output_file = loadpath_sobol + "Rank_" + gsa_mode + "_" + mode + ".txt"
    
    with open(output_file, "w") as f:
        for i in range(len(xlabels)):
            f.write(xlabels[ranked[i]] + "\t" + str(ranked_S[i]) + "\n")

    print('Normalising ranked sensitivity to compute explained variance...')
    ranked_S_norm = list(np.array(ranked_S) / sum(ranked_S))

    ranked_S_norm_cumulative = []
    for i in range(len(xlabels)):
        ranked_S_norm_cumulative.append(sum(ranked_S_norm[0:i + 1]))

    # Output the cumulative variance
    if output_file is None:
        output_file = loadpath_sobol + "Rank_" + gsa_mode + "_" + mode + "_ExpVariance.txt"
    else:
        output_file = output_file[:-4] + "_ExpVariance.txt"

    with open(output_file, "w") as f:
        for i in range(len(xlabels)):
            f.write(xlabels[ranked[i]] + "\t" + str(ranked_S_norm[i]) + "\t" + str(ranked_S_norm_cumulative[i]) + "\n")

    # Identify important parameters based on the cutoff threshold
    if important_params_idx_file is not None:
        idx_cutoff = np.where(np.array(ranked_S_norm_cumulative) > threshold_cutoff)[0][0]
        idx_param = ranked[range(idx_cutoff + 1)]
        np.savetxt(important_params_idx_file, idx_param, fmt="%g") 

    # -----------------------------------------------
    # Now repeat the process for each individual ylabel:
    # -----------------------------------------------
    print('Ranking parameters individually for each ylabel...')

    # print(f"{S=}")

    S = S[:, features]  # Filter S to only include the features we are interested in

    for ylabel_idx in range(S.shape[1]):  # Loop over each ylabel
        print(f"Processing ylabel {ylabel_idx + 1}/{S.shape[1]}...")

        # Get the sensitivity indices for the current ylabel
        S_label_total = S[:, ylabel_idx]
        
        ranked_label = np.argsort(S_label_total)
        ranked_label = ranked_label[::-1]  # Reverse to get highest to lowest
        ranked_S_label = S_label_total[ranked_label]

        # Sanitize ylabel for use in the filename
        sanitized_ylabel = sanitize_filename(ylabels[ylabel_idx])

        # Output the ranking for this label
        label_output_file = f"{loadpath_sobol}/Rank_{gsa_mode}_{mode}_{sanitized_ylabel}.txt"

        with open(label_output_file, "w") as f:
            for i in range(len(xlabels)):
                f.write(xlabels[ranked_label[i]] + "\t" + str(ranked_S_label[i]) + "\n")
        
        # Also compute and save cumulative variance for this ylabel
        ranked_S_label_norm = list(np.array(ranked_S_label) / sum(ranked_S_label))
        ranked_S_label_norm_cumulative = []
        for i in range(len(xlabels)):
            ranked_S_label_norm_cumulative.append(sum(ranked_S_label_norm[0:i + 1]))

        label_output_variance_file = label_output_file[:-4] + "_ExpVariance.txt"
        
        with open(label_output_variance_file, "w") as f:
            for i in range(len(xlabels)):
                f.write(xlabels[ranked_label[i]] + "\t" + str(ranked_S_label_norm[i]) + "\t" + str(ranked_S_label_norm_cumulative[i]) + "\n")

    print("Ranking completed for all labels.")

def plot_gsa_ranking_multiple_scenarios(scenarios,ylabels_raw_all,features_idx_list, xlabels, ylabels, savepath, fontsize, plot_barchart, figname_preffix, legend=[], colors=[], threshold=0, top_n=100):

    """
    Plots the GSA ranking for multiple scenarios.

    Args:
        - scenarios: list of scenario folders
        - xlabels: xlabels for the plot
        - ylabels: ylabels for the plot
        - savepath: path to save the figures
        - fontsize: font size for the plot
        - figname_preffix: prefix for the figure names
        - legend: legend labels
        - colors: colors for the plot
        - threshold: threshold for displaying xlabels
    """

    # ylabels_all = read_labels(ylabels)

    # ylabels_all_no_latex = read_labels(f"{scenarios[0]}/data/ylabels.txt")

    # features_idx_list = np.loadtxt(f"{scenarios[0]}/data/features_idx_list_gsa.txt", dtype=int)
    features_idx_list = list(np.array(features_idx_list, dtype=int))



    print(f"{features_idx_list=}")
    print(f"{ylabels=}")
    print(f"{len(ylabels)=}")

    ylabels_no_latex = [ylabels[i] for i in features_idx_list]

    # Create radar chart for each scenario
    for i, scenario in enumerate(scenarios):


        meshname = scenario.split("/")[-4]

        gsa_parameters_ranking_S_free_pathfile(
            loadpath=f"{scenario}/data/",
            loadpath_sobol=f"{scenario}/output/",
            xlabels=xlabels,
            gsa_mode="Si_total",
            mode="max",
            threshold_cutoff=0,
            output_file=None,
            features_file=None,
            important_params_idx_file=None,
            ylabels_raw_all=ylabels_raw_all,
            features=features_idx_list
        )

        if plot_barchart:

            os.makedirs(savepath, exist_ok=True)

            plot_rank_GSA_free_th_color(
                    datapath=f"{scenario}/data/",
                    loadpath=f"{scenario}/output/",
                    xlabels=xlabels,
                    rank_file=f"{scenario}/output/Rank_Si_total_max.txt",
                    criterion="Si_total",
                    mode="max",
                    figname=f"{savepath}/rank_max_{meshname}_all_features.png",
                    normalise=False,
                    annotate=False,
                    fontsize=fontsize,
                    separate_colors=True,
                    color_important=None,
                    title = f"Parameter ranking for {legend[i]} branch",
                    top_n=top_n,
                    acc_var_th=0.95
                )
            

            for ylabel_raw in ylabels_no_latex:

                ylabel = sanitize_filename(ylabel_raw)
                plot_rank_GSA_free_th_color(
                    datapath=f"{scenario}/data/",
                    loadpath=f"{scenario}/output/",
                    xlabels=xlabels,
                    rank_file=f"{scenario}/output/Rank_Si_total_max_{ylabel}.txt",
                    criterion="Si_total",
                    mode="max",
                    figname=f"{savepath}/rank_max_{meshname}_{ylabel}.png",
                    normalise=False,
                    annotate=True,
                    fontsize=fontsize,
                    separate_colors=True,
                    color_important=None,
                    top_n=top_n,
                    acc_var_th=0.95
                )

        # if plot_barchart:
        #     # List your PNG files
        #     image_files = [f"{savepath}/rank_max_{scenario.split('/')[-4]}_{ylabel}.png" for scenario in scenarios]
        #     # Open images
        #     images = [Image.open(f) for f in image_files]
        #     # Get total width and max height
        #     final_Width = max(img.width for img in images)
        #     final_height = sum(img.height for img in images)
        #     # Create new blank image with total width
        #     combined_image = Image.new('RGBA', (final_Width, final_height), (255, 255, 255, 0))
        #     y_offset = 0
        #     for img in images:
        #         combined_image.paste(img, (0, y_offset))
        #         y_offset += img.height
        #     # Save the combined image
        #     combined_image.save(f"{savepath}/combined_ranking_{ylabel}.png")
        #     print(f"Combined image saved to: {savepath}/combined_ranking_{ylabel}.png")
        #     # Close the images to free memory
        #     for img in images:
        #         img.close()
        
    if plot_barchart:
            # List your PNG filen
        image_files = [f"{savepath}/rank_max_{scenario.split('/')[-4]}_all_features.png" for scenario in scenarios]

        # Open images
        images = [Image.open(f) for f in image_files]

        # Get total width and max height
        final_Width = max(img.width for img in images)
        final_height = sum(img.height for img in images)

        # Create new blank image with total width
        combined_image = Image.new('RGBA', (final_Width, final_height), (255, 255, 255, 0))

        y_offset = 0
        for img in images:
            combined_image.paste(img, (0, y_offset))
            y_offset += img.height

        # Save the combined image
        combined_image.save(f"{savepath}/combined_ranking.png")

        print(f"Combined image saved to: {savepath}/combined_ranking.png")
        

def generate_gsa_visualizations(
    scenarios,
    xlabels_file,
    ylabels_file,
    savepath,
    plot_barchart,
    ylabels_dict,
    xlabels_dict,
    fontsize=14,
    figname_preffix="",
    legend=None,
    colors=None,
    threshold=0.0,
    plot_radar_chart=False,
    top_n=10,
    plot_bump_chart=False
):
    """
    Wrapper function to generate both GSA ranking and radar chart visualizations.

    Args:
        - scenarios: list of paths to the scenario folders
        - xlabels_file: path to the xlabels file
        - ylabels_file: path to the ylabels file
        - savepath: path to save the output figures
        - fontsize: font size for plots
        - figname_preffix: prefix for output figure names
        - legend: optional legend list
        - colors: optional list of colors
        - threshold: threshold to filter parameters in radar chart
    """

    if legend is None:
        legend = []

    if colors is None:
        colors = []

    xlabels = read_labels(xlabels_file)
    ylabels_all = read_labels(ylabels_file)

    # ylabels_all_no_latex = read_labels(f"{scenarios[0]}/data/ylabels.txt")

    ylabels_raw_all, ylabels_latex_all, features_idx_list = read_ylabels_dict(ylabels_dict, ylabels_all)

    print(f"Will process a total of {len(features_idx_list)} features.")

    xlabels_plain_all, xlabels_dict_all = read_xlabels_dict(xlabels_dict, xlabels)


    plot_gsa_ranking_multiple_scenarios(
        scenarios=scenarios,
        xlabels=[xlabels_dict_all[i]['latex'] if i in xlabels_dict_all else i for i in xlabels_plain_all],
        ylabels=ylabels_raw_all,
        savepath=savepath,
        fontsize=fontsize,
        figname_preffix=figname_preffix,
        legend=legend,
        colors=colors,
        threshold=threshold,
        plot_barchart = plot_barchart,
        features_idx_list=features_idx_list,
        ylabels_raw_all= ylabels_raw_all,
        top_n=top_n
    )


    if plot_bump_chart:
        plot_bump_chart_from_rankings_color_specific(
            scenarios=scenarios,
            savepath=savepath,
            legend=legend,
            xlabels_dict=xlabels_dict_all,
            fontsize=fontsize,
            gsa_mode="Si_total",
            mode="max",
            figname=f"{figname_preffix}_bump_chart_colored.png",
            rank_file=None,  # Use default rank file
            title="Bump chart of parameter rankings in different patients for all functional outputs",
            top_n=top_n
            )

        
        plot_bump_chart_from_rankings(
            scenarios=scenarios,
            savepath=savepath,
            fontsize=fontsize,
            gsa_mode="Si_total",
            mode="max",
            figname=f"{figname_preffix}_bump_chart.png",
            title="Bump chart of parameter rankings in different patients for all functional outputs",
            legend=legend,
            xlabels_to_plot=xlabels_dict_all,
            top_n=top_n
            )
    
    # features_idx_list = np.loadtxt(f"{scenarios[0]}/data/features_idx_list_gsa.txt", dtype=int)
    # if len(features_idx_list.shape) == 0:
    #     features_idx_list = [features_idx_list]
    # else:
    #     features_idx_list = list(features_idx_list)

    # ylabels_no_latex = ylabels_all_no_latex[features_idx_list]

    # ylabels_no_latex = [ylabels_raw_all[i] for i in features_idx_list]

    for loop_idx, feature_idx in enumerate(features_idx_list):

        if plot_bump_chart:
            plot_bump_chart_from_rankings_color_specific(
                scenarios=scenarios,
                savepath=savepath,
                legend=legend,
                xlabels_dict=xlabels_dict_all,
                fontsize=fontsize,
                gsa_mode="Si_total",
                mode="max",
                figname=f"{figname_preffix}_bump_chart_{ylabels_raw_all[feature_idx]}_colored.png",
                rank_file=f"Rank_Si_total_max_{ylabels_raw_all[feature_idx]}.txt",
                title=f"Bump chart of parameter rankings in different patients for {ylabels_latex_all[feature_idx]}",
                top_n=top_n
            )

            
            plot_bump_chart_from_rankings(
                scenarios=scenarios,
                savepath=savepath,
                fontsize=fontsize,
                gsa_mode="Si_total",
                mode="max",
                figname=f"{figname_preffix}_bump_chart_{ylabels_raw_all[feature_idx]}.png",
                rank_file=f"Rank_Si_total_max_{ylabels_raw_all[feature_idx]}.txt",
                title=f"Bump chart of parameter rankings in different patients for {ylabels_latex_all[feature_idx]}",
                legend=legend,
                xlabels_to_plot=xlabels_dict_all,
                top_n=top_n
            )

    
        if plot_radar_chart:
            plot_GSA_radar_chart_feature(
                scenarios=scenarios,
                xlabels_plain_all=xlabels_plain_all,
                xlabels_dict_all=xlabels_dict_all,
                savepath=savepath,
                fontsize=fontsize,
                figname_preffix=figname_preffix,
                legend=legend,
                colors=colors,
                threshold=threshold,
                feature_idx=feature_idx,
                loop_idx=loop_idx,
                ylabels_latex_all=ylabels_latex_all,
                ylabels_raw_all=ylabels_raw_all
            )



def read_ylabels_dict(ylabels_dict_file, ylabels_all):

    print(f"Reading {ylabels_dict_file}")
    with open(ylabels_dict_file, 'r') as f:
        ylabels_dict = json.load(f)

    features_idx_list = []
    ylabels_raw_all = []
    ylabels_latex_all = []

    for label in ylabels_dict.keys():
        ylabels_latex_all.append(ylabels_dict[label]["latex"])
        ylabels_raw_all.append(label)
        if ylabels_dict[label]["run"] == 1:
            # We find where in ylabels_all is label:
            if label in ylabels_all:
                idx = np.where(ylabels_all == label)[0][0]
                features_idx_list.append(idx)
            else:
                raise ValueError(f"Label '{label}' not found in ylabels_all. {ylabels_all=}")
            
    return ylabels_raw_all, ylabels_latex_all, features_idx_list

def read_xlabels_dict(xlabels_dict_file, xlabels_all):
    
    with open(xlabels_dict_file, 'r') as f:
        xlabels_dict = json.load(f)

    return xlabels_all, xlabels_dict
    
def plot_bump_chart_from_rankings_color_specific(
    scenarios,
    savepath,
    legend,
    xlabels_dict,
    fontsize=12,
    gsa_mode="Si_total",
    mode="max",
    figname="bump_chart.png",
    rank_file=None,
    title=None,
    top_n=40
):
    """
    Generates a bump chart showing parameter rankings with parameter-specific colors.

    - Top_n parameters in scenario 1 have continuous lines and points.
    - Parameters outside top_n in scenario 1 but in top_n later appear with unique markers.
    - Colors are assigned per parameter using xlabels_dict[param]["color"].
    - Low-importance parameters (effect < 0.05) are shown with transparent colors.
    - Parameters outside top_n are shown in a legend if they appear in top_n of any scenario.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgba
    from collections import defaultdict

    param_ranks = defaultdict(dict)
    param_effects = defaultdict(dict)
    all_params = set()
    param_names_per_scenario = []

    if rank_file is None:
        rank_file = f"Rank_{gsa_mode}_{mode}.txt"

    # ------------------------
    # Load rankings and effects
    # ------------------------
    for scenario in scenarios:
        rank_file_full_path = os.path.join(scenario, "output", rank_file)
        scenario_name = scenario.rstrip("/").split("/")[-3]

        with open(rank_file_full_path, "r") as f:
            params_in_this_scenario = []
            for i, line in enumerate(f.readlines()):
                param, value = line.strip().split("\t")
                param_ranks[param][scenario_name] = i + 1
                param_effects[param][scenario_name] = float(value)
                all_params.add(param)
                params_in_this_scenario.append(param)
            param_names_per_scenario.append(params_in_this_scenario)

    # ------------------------
    # Consistency check
    # ------------------------
    first_param_set = set(param_names_per_scenario[0])
    first_rank_file = os.path.join(scenarios[0], "output", rank_file)
    for idx, param_list in enumerate(param_names_per_scenario[1:], start=1):
        this_param_set = set(param_list)
        this_rank_file = os.path.join(scenarios[idx], "output", rank_file)
        missing = first_param_set - this_param_set
        extra = this_param_set - first_param_set
        if missing or extra:
            print(f"\nParameter names mismatch detected!")
            print(f"First scenario rank file: {first_rank_file}")
            print(f"Current scenario rank file: {this_rank_file}")
            if missing:
                print(f"Parameters missing in scenario {idx+1}: {sorted(missing)}")
            if extra:
                print(f"Extra parameters in scenario {idx+1}: {sorted(extra)}")
            raise ValueError("Parameter names mismatch between scenarios.")

    if len(all_params) != len(param_names_per_scenario[0]):
        raise ValueError("Possible repeated parameter names in the xlabels file.")

    all_scenarios = [s.rstrip("/").split("/")[-3] for s in scenarios]
    scenario1, last_scenario = all_scenarios[0], all_scenarios[-1]

    # ------------------------
    # Identify top_n parameters for first and last scenario
    # ------------------------
    top_n_params = sorted(first_param_set, key=lambda p: param_ranks[p][scenario1])[:top_n]
    top_n_last_params = sorted(first_param_set, key=lambda p: param_ranks[p][last_scenario])[:top_n]
    outside_top_n_params = sorted(all_params - set(top_n_params))

    # ------------------------
    # Assign markers to outside params
    # ------------------------
    special_markers = ['s', 'D', '^', 'v', 'P', '*', 'X', 'h', '<', '>']
    param_to_marker = {
        param: special_markers[i % len(special_markers)]
        for i, param in enumerate(outside_top_n_params)
    }

    # ------------------------
    # Initialize plot
    # ------------------------
    fig_width = 10
    fig_height = max(6, len(top_n_params) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=False)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)

    outside_params_plotted = set()

    # ------------------------
    # Helper functions
    # ------------------------
    def appears_in_top_n_any_scenario(param):
        return any(param_ranks[param].get(s, len(all_params)+1) <= top_n for s in all_scenarios)

    def get_color_for_param(param):
        for k, v in xlabels_dict.items():
            if v.get("latex") == param:
                return v.get("color", "#8F8F8F")
        return "#8F8F8F"

    def get_color_with_alpha(base_color, effect):
        alpha = 0.3 if effect < 0.05 else 1.0
        return to_rgba(base_color, alpha)

    # ------------------------
    # Plot top_n parameters
    # ------------------------
    for param in top_n_params:
        ranks = [param_ranks[param].get(s, len(all_params)+1) for s in all_scenarios]
        effects = [param_effects[param].get(s, 0.0) for s in all_scenarios]
        ranks_clipped = [r if r <= top_n else np.nan for r in ranks]
        base_color = get_color_for_param(param)

        for i in range(len(all_scenarios)-1):
            r1, r2 = ranks_clipped[i], ranks_clipped[i+1]
            if np.isnan(r1) or np.isnan(r2):
                continue
            effect = max(effects[i], effects[i+1])
            color = get_color_with_alpha(base_color, effect)
            ax.plot([i, i+1], [r1, r2], color=color, linewidth=2, zorder=1)

        for ix, (rank, effect) in enumerate(zip(ranks_clipped, effects)):
            if np.isnan(rank):
                continue
            color = get_color_with_alpha(base_color, effect)
            ax.scatter(ix, rank, s=200, marker='.', color=color, zorder=3)

    # ------------------------
    # Plot outside_top_n parameters
    # ------------------------
    for param in outside_top_n_params:
        ranks_all = [param_ranks[param].get(s, len(all_params)+1) for s in all_scenarios]
        effects_all = [param_effects[param].get(s, 0.0) for s in all_scenarios]
        marker = param_to_marker[param]
        base_color = get_color_for_param(param)
        prev_rank, prev_ix = None, None
        appeared_in_top_n = False

        for ix, (rank, effect) in enumerate(zip(ranks_all, effects_all)):
            color = get_color_with_alpha(base_color, effect)
            if rank <= top_n:
                appeared_in_top_n = True
                ax.scatter(ix, rank, s=100, marker=marker,
                           facecolors='white', edgecolors=color,
                           linewidths=1.5, zorder=3)

                if prev_rank is not None and prev_rank <= top_n:
                    ax.plot([prev_ix, ix], [prev_rank, rank],
                            color=color, linewidth=1.5, zorder=2)

            prev_rank, prev_ix = rank, ix

        if appeared_in_top_n:
            outside_params_plotted.add(param)

    # ------------------------
    # Label parameters
    # ------------------------
    for param in top_n_params:
        rank1 = param_ranks[param][scenario1]
        label = xlabels_dict.get(param, {}).get("latex", param)
        ax.text(-0.3, rank1, label, fontsize=fontsize-2, va='center', ha='right')

    for param in top_n_last_params:
        rank_last = param_ranks[param][last_scenario]
        label = xlabels_dict.get(param, {}).get("latex", param)
        ax.text(len(all_scenarios)-0.7, rank_last, label, fontsize=fontsize-2, va='center', ha='left')

    # ------------------------
    # Axis and title formatting
    # ------------------------
    ax.invert_yaxis()
    ax.set_ylim(top_n+0.5, 0.5)
    ax.set_xticks(np.arange(len(all_scenarios)))
    ax.set_xticklabels(legend, fontsize=fontsize, rotation=30)
    ax.axes.yaxis.set_visible(False)
    ax.spines[:].set_visible(False)

    if title is None:
        title = "Bump Chart of Parameter Rankings in Different Meshes"
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold')

    # ------------------------
    # Legend for parameter categories
    # ------------------------
    legend_mapping = {
        "Rsys": "Left side parameter",
        "Rpulm": "Right side parameter",
        "a_ventricles": "Either side parameter"
    }

    legend_handles = []
    for param, label in legend_mapping.items():
        color = xlabels_dict.get(param, {}).get("color", "#8F8F8F")
        handle = Line2D([0], [0], color=color, lw=3, label=label)
        legend_handles.append(handle)

    ax.legend(
        handles=legend_handles,
        loc='upper center',
        fontsize=fontsize-1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.4),
        ncol=3
    )

    # ------------------------
    # Legend for outside top_n parameters
    # ------------------------
    outside_legend_params = (outside_params_plotted - set(top_n_last_params))
    if outside_legend_params:
        special_legend_handles = []
        for param in sorted(outside_legend_params):
            marker = param_to_marker[param]
            base_color = get_color_for_param(param)
            effects = [param_effects[param].get(s, 0.0) for s in all_scenarios]
            avg_effect = np.mean(effects)
            color = get_color_with_alpha(base_color, avg_effect)
            label = xlabels_dict.get(param, {}).get("latex", param)

            handle = Line2D(
                [0], [0],
                marker=marker,
                color='black',
                label=label,
                markerfacecolor='white',
                markeredgecolor=color,
                markersize=10,
                linestyle='None',
                markeredgewidth=1.5
            )
            special_legend_handles.append(handle)

        ax.figure.legend(
            handles=special_legend_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.2),
            fontsize=fontsize-2,
            frameon=False,
            ncol=min(len(special_legend_handles), 5)
        )

    # ------------------------
    # Save
    # ------------------------
    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Bump chart saved to: {output_path}")

def plot_bump_chart_from_rankings(
    scenarios,
    savepath,
    legend,
    fontsize=12,
    gsa_mode="Si_total",
    mode="max",
    figname="bump_chart.png",
    rank_file=None,
    title=None,
    top_n=40,
    xlabels_to_plot=None
):
    """
    Generates a bump chart showing parameter rankings with simplified coloring.
    Coloring logic:
        - Low importance: effect < 0.05
        - High importance & stable: effect ≥ 0.05 and small variability
        - High importance & highly variable: effect ≥ 0.05 and rank_range ≥ 10
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from collections import defaultdict

    if xlabels_to_plot is None:
        xlabels_to_plot = {}

    param_ranks = defaultdict(dict)
    param_effects = defaultdict(dict)
    all_params = set()
    param_names_per_scenario = []

    if rank_file is None:
        rank_file = f"Rank_{gsa_mode}_{mode}.txt"

    # ------------------------
    # Load rankings and effects
    # ------------------------
    for scenario in scenarios:
        rank_file_full_path = os.path.join(scenario, "output", rank_file)
        scenario_name = scenario.rstrip("/").split("/")[-3]

        with open(rank_file_full_path, "r") as f:
            params_in_this_scenario = []
            for i, line in enumerate(f.readlines()):
                param, value = line.strip().split("\t")
                param_ranks[param][scenario_name] = i + 1  # 1-based rank
                param_effects[param][scenario_name] = float(value)
                all_params.add(param)
                params_in_this_scenario.append(param)
            param_names_per_scenario.append(params_in_this_scenario)

    # ------------------------
    # Consistency check
    # ------------------------
    first_param_set = set(param_names_per_scenario[0])
    first_rank_file = os.path.join(scenarios[0], "output", rank_file)
    for idx, param_list in enumerate(param_names_per_scenario[1:], start=1):
        this_param_set = set(param_list)
        this_rank_file = os.path.join(scenarios[idx], "output", rank_file)
        missing = first_param_set - this_param_set
        extra = this_param_set - first_param_set
        if missing or extra:
            print(f"\nParameter names mismatch detected!")
            print(f"First scenario rank file: {first_rank_file}")
            print(f"Current scenario rank file: {this_rank_file}")
            if missing:
                print(f"Parameters missing in scenario {idx+1}: {sorted(missing)}")
            if extra:
                print(f"Extra parameters in scenario {idx+1}: {sorted(extra)}")
            raise ValueError("Parameter names mismatch between scenarios.")

    if len(all_params) != len(param_names_per_scenario[0]):
        raise ValueError("Possible repeated parameter names in the xlabels file.")

    all_scenarios = [s.rstrip("/").split("/")[-3] for s in scenarios]
    scenario1, last_scenario = all_scenarios[0], all_scenarios[-1]

    # ------------------------
    # Identify top_n parameters for first and last scenario
    # ------------------------
    top_n_params = sorted(first_param_set, key=lambda p: param_ranks[p][scenario1])[:top_n]
    top_n_last_params = sorted(first_param_set, key=lambda p: param_ranks[p][last_scenario])[:top_n]
    outside_top_n_params = sorted(all_params - set(top_n_params))

    # ------------------------
    # Assign markers to outside params
    # ------------------------
    special_markers = ['s', 'D', '^', 'v', 'P', '*', 'X', 'h', '<', '>']
    param_to_marker = {
        param: special_markers[i % len(special_markers)]
        for i, param in enumerate(outside_top_n_params)
    }

    # ------------------------
    # Initialize plot
    # ------------------------
    fig, ax = plt.subplots(figsize=(12, max(6, len(top_n_params) * 0.25)), constrained_layout=True)
    very_light_grey = "#8888887A"
    grey = "#888888"
    high_variability = "#c08978"
    colors_used_top_n, outside_params_plotted = set(), set()

    # Helper function: determine color
    def get_color(effect, rank_range):
        if effect < 0.05:
            return very_light_grey, 'low_importance'
        elif rank_range >= 10:
            return high_variability, 'high_variability'
        else:
            return grey, 'high_importance'

    # ------------------------
    # Track high variability parameters across all parameters (not just top_n)
    # ------------------------
    high_var_params_all = set()
    for param in all_params:
        ranks = [param_ranks[param].get(s, len(all_params)+1) for s in all_scenarios]
        effects = [param_effects[param].get(s, 0.0) for s in all_scenarios]
        rank_range = max(ranks) - min(ranks)
        if max(effects) >= 0.05 and rank_range >= 10:
            high_var_params_all.add(param)

    # ------------------------
    # Plot top_n parameters (from first scenario)
    # ------------------------
    for param in top_n_params:
        ranks = [param_ranks[param].get(s, len(all_params)+1) for s in all_scenarios]
        effects = [param_effects[param].get(s, 0.0) for s in all_scenarios]
        ranks_clipped = [r if r <= top_n else np.nan for r in ranks]
        rank_range = max(ranks) - min(ranks)

        # Lines
        for i in range(len(all_scenarios)-1):
            r1, r2 = ranks_clipped[i], ranks_clipped[i+1]
            if np.isnan(r1) or np.isnan(r2):
                continue
            color, color_label = get_color(max(effects[i], effects[i+1]), rank_range)
            colors_used_top_n.add(color_label)  # Track colors used for legend
            ax.plot([i, i+1], [r1, r2], color=color, linewidth=2, zorder=1)

        # Points
        for ix, (rank, effect) in enumerate(zip(ranks_clipped, effects)):
            if np.isnan(rank):
                continue
            color, color_label = get_color(effect, rank_range)
            colors_used_top_n.add(color_label)  # Track colors used for legend
            ax.scatter(ix, rank, s=200, marker='.', color=color, zorder=3)

    # ------------------------
    # Helper: check if param appears in top_n of any scenario
    # ------------------------
    def appears_in_top_n_any_scenario(param):
        for s in all_scenarios:
            if param_ranks[param].get(s, len(all_params)+1) <= top_n:
                return True
        return False

    # ------------------------
    # Plot outside_top_n parameters and track those for legend
    # ------------------------
    high_var_params_for_legend = set()

    for param in outside_top_n_params:
        ranks_all = [param_ranks[param].get(s, len(all_params)+1) for s in all_scenarios]
        effects_all = [param_effects[param].get(s, 0.0) for s in all_scenarios]
        rank_range = max(ranks_all) - min(ranks_all)
        marker = param_to_marker[param]
        color, _ = get_color(max(effects_all), rank_range)

        prev_ix, prev_rank = None, None
        appeared_in_top_n = False
        for ix, rank in enumerate(ranks_all):
            if rank <= top_n:
                appeared_in_top_n = True
                ax.scatter(ix, rank, s=100, marker=marker, facecolors='white',
                           edgecolors=color, linewidths=1.5, zorder=3)
                if prev_ix is not None:
                    ax.plot([prev_ix, ix], [prev_rank, rank], color=color,
                            linewidth=1.5, zorder=1, linestyle='-')
                prev_ix, prev_rank = ix, rank
                outside_params_plotted.add(param)
            else:
                prev_ix, prev_rank = None, None

        # Add to legend if highly variable AND appears in top_n of any scenario
        if param in high_var_params_all and appears_in_top_n_any_scenario(param):
            high_var_params_for_legend.add(param)

    # ------------------------
    # Label parameters on left (first scenario) and right (last scenario)
    # ------------------------
    for param in top_n_params:
        rank1 = param_ranks[param][scenario1]
        label = xlabels_to_plot.get(param, {}).get("latex", param)
        ax.text(-0.3, rank1, label, fontsize=fontsize-2, va='center', ha='right')

    for param in top_n_last_params:
        rank_last = param_ranks[param][last_scenario]
        label = xlabels_to_plot.get(param, {}).get("latex", param)
        ax.text(len(all_scenarios)-0.7, rank_last, label, fontsize=fontsize-2, va='center', ha='left')

    # ------------------------
    # Axis & title formatting
    # ------------------------
    ax.invert_yaxis()
    ax.set_ylim(top_n+0.5, 0.5)
    ax.set_xticks(np.arange(len(all_scenarios)))
    ax.set_xticklabels(legend, fontsize=fontsize, rotation=30)
    ax.axes.yaxis.set_visible(False)
    ax.spines[:].set_visible(False)
    ax.set_title(title or "Bump Chart of Parameter Rankings in Different Meshes",
                 fontsize=fontsize+2, fontweight='bold')

    # ------------------------
    # Legends
    # ------------------------
    legend_elements = []
    if 'low_importance' in colors_used_top_n:
        legend_elements.append(Line2D([0], [0], color=very_light_grey, lw=2, label='Low importance'))
    if 'high_importance' in colors_used_top_n and 'low_importance' in colors_used_top_n:
        legend_elements.append(Line2D([0], [0], color=grey, lw=2, label='High importance'))
    # Show high variability legend if any high variability parameter exists in the plot
    if high_var_params_all and 'low_importance' in colors_used_top_n:
        legend_elements.append(Line2D([0], [0], color=high_variability, lw=2,
                                      label='High importance & highly variable'))
    if high_var_params_all and 'low_importance' not in colors_used_top_n:
        legend_elements.append(Line2D([0], [0], color=high_variability, lw=2,
                                      label='Highly variable'))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, -0.4), fontsize=fontsize-2, frameon=False, ncol=3)

    # Combine outside params that were plotted + high variability params to show in legend,
    # excluding those already labeled on right side in last scenario top_n
    outside_legend_params = (outside_params_plotted | high_var_params_for_legend) - set(top_n_last_params)
    if outside_legend_params:
        special_legend_handles = [
            Line2D([0], [0], marker=param_to_marker[param], color='black',
                   label=xlabels_to_plot.get(param, {}).get("latex", param),
                   markerfacecolor='white', markersize=10, linestyle='None', markeredgewidth=1.5)
            for param in sorted(outside_legend_params)
        ]
        ax.figure.legend(handles=special_legend_handles, loc='lower center',
                         bbox_to_anchor=(0.5, -0.2), fontsize=fontsize-2,
                         frameon=False, ncol=min(len(special_legend_handles), 5))

    # ------------------------
    # Save
    # ------------------------
    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Bump chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot GSA Visualizations")
    parser.add_argument('--scenarios', nargs='+', required=True, help='Paths to the scenario folders')
    parser.add_argument('--xlabels', required=True, help='Path to the xlabels file')
    parser.add_argument('--ylabels', required=True, help='Path to the ylabels file')
    parser.add_argument('--savepath', required=True, help='Path to save the figures')
    parser.add_argument('--fontsize', type=int, default=14, help='Font size for the plot')
    parser.add_argument('--figname_preffix', required=True, help='Prefix for the figure names')
    parser.add_argument('--legend', nargs='*', default=[], help='Legend labels')
    parser.add_argument('--colors', nargs='*', default=[], help='Colors for the plot')
    parser.add_argument('--threshold', type=float, default=0, help='Threshold for displaying xlabels')
    parser.add_argument('--plot_barchart', action='store_true', help='Whether to plot the GSA ranking as a bar chart')
    parser.add_argument('--plot_radar_chart', action='store_true', help='Whether to plot the GSA ranking as a radar chart')
    parser.add_argument('--plot_bump_chart', action='store_true', help='Whether to plot the GSA ranking as a bump chart')
    parser.add_argument('--ylabels_dict', type=str, required=True, help='Path to the ylabels dictionary file (optional)')
    parser.add_argument('--xlabels_dict', type=str, required=True, help='Path to the xlabels dictionary file (optional)')
    parser.add_argument('--top_n', type=int, default=100, help='Number of top parameters to display in the bump chart')
    args = parser.parse_args()


    generate_gsa_visualizations(
        scenarios=args.scenarios,
        xlabels_file=args.xlabels,
        ylabels_file=args.ylabels,
        savepath=args.savepath,
        fontsize=args.fontsize,
        figname_preffix=args.figname_preffix,
        legend=args.legend,
        colors=args.colors,
        threshold=args.threshold,
        plot_barchart=args.plot_barchart,
        plot_radar_chart=args.plot_radar_chart,
        plot_bump_chart=args.plot_bump_chart,
        ylabels_dict=args.ylabels_dict,
        xlabels_dict=args.xlabels_dict,
        top_n=args.top_n
    )

if __name__ == "__main__":
    main()