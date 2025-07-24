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

def plot_GSA_radar_chart_feature(scenarios, xlabels, ylabels_latex_all, ylabels_raw_all, savepath, fontsize, figname_preffix, legend=[], 
                                 colors=[], threshold=0, feature_idx=0, loop_idx=0):
    

    output_idx = feature_idx
    # ylabel_name = ylabels[output_idx]

    # Sanitize the filename to ensure it is valid
    # sanitized_ylabel_name = sanitize_filename(ylabel_name)
    # figname = f"{figname_preffix}_{sanitized_ylabel_name}"

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
                                color_all="#ff8000"):
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

    r = [xx + barWidth for xx in x]
    
    
    ax.set_xlim(min(r) - barWidth, max(r) + barWidth)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    bars_sorted = [bars[idx] for idx in idx_sorted]
    bars_sorted = bars_sorted[::-1]
    
    index_i_sorted = [index_i[idx] for idx in idx_sorted]
    index_i_sorted = index_i_sorted[::-1]

    bars_sorted_norm = list(np.array(bars_sorted) / sum(bars_sorted))

    bars_sorted_sum = []
    for i in range(len(bars_sorted)):
        bars_sorted_sum.append(sum(bars_sorted_norm[0:i+1]))

    if normalise:
        barplot = bars_sorted_norm
    else:
        barplot = bars_sorted

    plt.xticks(x + barWidth, index_i_sorted, rotation=90, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    cutoff_param = np.where(np.array(bars_sorted_sum) > acc_var_th)[0][0]
    if color_important is None:
        color_important = "#387339"
    color_unimportant = "lightgray"
    colors = [color_important] * (cutoff_param + 1) + [color_unimportant] * (len(bars_sorted_sum) - cutoff_param - 1)

    if separate_colors:
        bars = ax.bar(r, barplot, width=barWidth, edgecolor='white', color=colors)
    else:
        bars = ax.bar(r, barplot, color=color, width=barWidth, edgecolor='white')

    if criterion == 'Si_total':
        ax.set_ylabel('Total-order effects', fontsize=fontsize)
    else:
        ax.set_ylabel('First-order effects', fontsize=fontsize)
    
    # if th > 0:
    #     ax.plot([-2 * barWidth, len(index_i) + 2 * barWidth], [th, th], color='black', linestyle='--')
    
    # plt.legend()

    meshname = loadpath.split("/")[-6]

    ax.set_title(f"Parameter ranking for mesh #{meshname}", fontsize=fontsize + 5, fontweight="bold")

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

    print(f"{S=}")

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

def plot_gsa_ranking_multiple_scenarios(scenarios,ylabels_raw_all,features_idx_list, xlabels, ylabels, savepath, fontsize, plot_barchart, figname_preffix, legend=[], colors=[], threshold=0):

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
                    annotate=True,
                    fontsize=fontsize,
                    separate_colors=True,
                    color_important=None,
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
    plot_radar_chart=False
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

    plot_gsa_ranking_multiple_scenarios(
        scenarios=scenarios,
        xlabels=xlabels,
        ylabels=ylabels_raw_all,
        savepath=savepath,
        fontsize=fontsize,
        figname_preffix=figname_preffix,
        legend=legend,
        colors=colors,
        threshold=threshold,
        plot_barchart = plot_barchart,
        features_idx_list=features_idx_list,
        ylabels_raw_all= ylabels_raw_all
    )

    xlabels_latex_all, xlabels_dict = read_xlabels_dict(xlabels_dict, xlabels)
    
    plot_bump_chart_from_rankings_color_specific(
        scenarios=scenarios,
        savepath=savepath,
        legend=legend,
        xlabels_dict=xlabels_dict,
        fontsize=fontsize,
        gsa_mode="Si_total",
        mode="max",
        figname=f"{figname_preffix}_bump_chart_colored.png",
        rank_file=None,  # Use default rank file
        title="Bump chart of parameter rankings in different patients for all functional outputs",
    )

    """
    plot_bump_chart_from_rankings(
        scenarios=scenarios,
        savepath=savepath,
        fontsize=fontsize,
        gsa_mode="Si_total",
        mode="max",
        figname=f"{figname_preffix}_bump_chart.png",
        title="Bump chart of parameter rankings in different patients for all functional outputs",
        legend=legend,
        xlabels_latex_all=xlabels_latex_all
    )
    """
    # features_idx_list = np.loadtxt(f"{scenarios[0]}/data/features_idx_list_gsa.txt", dtype=int)
    # if len(features_idx_list.shape) == 0:
    #     features_idx_list = [features_idx_list]
    # else:
    #     features_idx_list = list(features_idx_list)

    # ylabels_no_latex = ylabels_all_no_latex[features_idx_list]

    # ylabels_no_latex = [ylabels_raw_all[i] for i in features_idx_list]

    for loop_idx, feature_idx in enumerate(features_idx_list):

        plot_bump_chart_from_rankings_color_specific(
            scenarios=scenarios,
            savepath=savepath,
            legend=legend,
            xlabels_dict=xlabels_dict,
            fontsize=fontsize,
            gsa_mode="Si_total",
            mode="max",
            figname=f"{figname_preffix}_bump_chart_{ylabels_raw_all[feature_idx]}_colored.png",
            rank_file=f"Rank_Si_total_max_{ylabels_raw_all[feature_idx]}.txt",
            title=f"Bump chart of parameter rankings in different patients for {ylabels_latex_all[feature_idx]}",
        )

        """ 
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
            xlabels_latex_all=xlabels_latex_all
        )

    """
        if plot_radar_chart:
            plot_GSA_radar_chart_feature(
                scenarios=scenarios,
                xlabels=xlabels,
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
    scenarios, savepath, legend, xlabels_dict,
    fontsize=12, gsa_mode="Si_total", mode="max",
    figname="bump_chart.png", rank_file=None, title=None
):
    """
    Generates a bump chart from multiple scenario rankings with parameter-based color coding.

    Args:
        - scenarios: list of scenario paths (each must have a rank file inside output/)
        - savepath: directory to save the output chart
        - legend: list of x-axis labels (e.g., scenario names)
        - xlabels_dict: dict mapping param names to {"latex": ..., "color": ...}
        - fontsize: font size for plot labels
        - gsa_mode: "Si_total" or "Si", depending on which ranking file was generated
        - mode: ranking strategy: "max", "mean", or "sum"
        - figname: filename of the output bump chart
    """

    param_ranks = defaultdict(dict)
    param_effects = defaultdict(dict)
    all_params = set()
    param_names_per_scenario = []

    if rank_file is None:
        rank_file = f"Rank_{gsa_mode}_{mode}.txt"

    # Step 1: Read rankings and effects
    for scenario in scenarios:
        rank_file_full_path = os.path.join(scenario, "output", rank_file)
        scenario_name = scenario.rstrip("/").split("/")[-3]  # Adjust if needed

        with open(rank_file_full_path, "r") as f:
            params_in_this_scenario = []
            for i, line in enumerate(f.readlines()):
                param, value = line.strip().split("\t")
                param_ranks[param][scenario_name] = i + 1
                param_effects[param][scenario_name] = float(value)
                all_params.add(param)
                params_in_this_scenario.append(param)
            param_names_per_scenario.append(params_in_this_scenario)

    # Check consistency across scenarios
    base_set = set(param_names_per_scenario[0])
    for idx, param_list in enumerate(param_names_per_scenario[1:], start=1):
        this_set = set(param_list)
        missing = base_set - this_set
        extra = this_set - base_set
        if missing or extra:
            raise ValueError(f"Parameter mismatch in scenario {idx + 1}")

    all_params = sorted(all_params)
    all_scenarios = [s.rstrip("/").split("/")[-3] for s in scenarios]

    # Step 2: Convert to rank matrix
    rank_matrix = []
    for param in all_params:
        rank_matrix.append([
            param_ranks[param].get(scen, len(all_params) + 1)
            for scen in all_scenarios
        ])

    # Step 3: Plot
    plt.figure(figsize=(12, max(6, len(all_params) * 0.25)))
    ax = plt.gca()

    for idx, (param, ranks) in enumerate(zip(all_params, rank_matrix)):
        effects = [param_effects[param].get(scen, 0.0) for scen in all_scenarios]
        rank_range = max(ranks) - min(ranks)

        x = np.arange(len(all_scenarios))
        y = np.array(ranks)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Match param to the correct xlabels_dict entry by comparing latex strings
        base_color = "#8F8F8F"  # default light grey
        for entry in xlabels_dict.values():
            if entry.get("latex") == param:
                c = entry.get("color", "").strip()
                if c:
                    base_color = c
                break

        # Draw line segments
        for i in range(len(segments)):
            e1, e2 = effects[i], effects[i+1]
            segment_color = to_rgba(base_color, alpha=0.25 if e1 < 0.05 and e2 < 0.05 else 1.0)
            lc = LineCollection([segments[i]], colors=[segment_color], linewidths=2)
            ax.add_collection(lc)

        # Draw individual points
        for xi, yi, ei in zip(x, y, effects):
            point_color = to_rgba(base_color, alpha=0.25 if ei < 0.05 else 1.0)
            ax.scatter(xi, yi, color=point_color, zorder=3)

        # Add label to the left
        ax.text(
            -0.1, y[0], param,
            fontsize=fontsize - 3,
            va='center',
            ha='right'
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.invert_yaxis()
    ax.axes.yaxis.set_visible(False)

    if title is None:
        title = "Bump Chart of Parameter Rankings in Different Meshes"
    plt.title(title, fontsize=fontsize + 2, fontweight='bold')
    plt.xticks(ticks=np.arange(len(all_scenarios)), labels=legend, fontsize=fontsize, rotation=30)
    plt.tight_layout()

    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Bump chart saved to: {output_path}")
        

def plot_bump_chart_from_rankings(scenarios, savepath, legend, fontsize=12, gsa_mode="Si_total", mode="max", figname="bump_chart.png", rank_file=None, title=None):
    """
    Generates a bump chart from multiple scenario rankings to show how parameter importance changes.

    Args:
        - scenarios: list of scenario paths (each must have a rank file inside output/)
        - savepath: directory to save the output chart
        - fontsize: font size for plot labels
        - gsa_mode: "Si_total" or "Si", depending on which ranking file was generated
        - mode: ranking strategy: "max", "mean", or "sum"
        - figname: filename of the output bump chart
    """

    param_ranks = defaultdict(dict)
    param_effects = defaultdict(dict)
    all_params = set()
    param_names_per_scenario = []

    if rank_file is None:
        rank_file = f"Rank_{gsa_mode}_{mode}.txt"

    # Step 1: Collect all ranks and effects from each scenario
    for scenario in scenarios:
        rank_file_full_path = os.path.join(scenario, "output", rank_file)
        scenario_name = scenario.rstrip("/").split("/")[-3]  # Adjust this if needed

        with open(rank_file_full_path, "r") as f:
            params_in_this_scenario = []
            for i, line in enumerate(f.readlines()):
                param, value = line.strip().split("\t")
                param_ranks[param][scenario_name] = i + 1  # 1-based rank
                param_effects[param][scenario_name] = float(value)
                all_params.add(param)
                params_in_this_scenario.append(param)
            param_names_per_scenario.append(params_in_this_scenario)

    # Consistency check for parameter names across scenarios
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

    all_params = sorted(all_params)
    all_scenarios = [s.rstrip("/").split("/")[-3] for s in scenarios]

    # Step 2: Convert to 2D matrix of ranks
    rank_matrix = []
    for param in all_params:
        ranks = []
        for scen in all_scenarios:
            rank = param_ranks[param].get(scen, len(all_params) + 1)  # Unranked params to bottom
            ranks.append(rank)
        rank_matrix.append(ranks)

    # Step 3: Plot
    plt.figure(figsize=(12, max(6, len(all_params) * 0.25)))
    ax = plt.gca()

    for idx, (param, ranks) in enumerate(zip(all_params, rank_matrix)):
        effects = [param_effects[param].get(scen, 0.0) for scen in all_scenarios]
        rank_range = max(ranks) - min(ranks)

        x = np.arange(len(all_scenarios))
        y = np.array(ranks)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        segment_colors = []
        for i in range(len(segments)):
            e1, e2 = effects[i], effects[i+1]
            # Colors
            very_light_grey = "#8888887A"  # low importance
            grey = '#888888'             # high importance
            red = '#FF6961'
            # Both points low importance
            if e1 < 0.05 and e2 < 0.05:
                segment_colors.append(very_light_grey)
            # Both points high importance
            elif e1 >= 0.05 and e2 >= 0.05:
                if rank_range >= 10:
                    segment_colors.append(red)
                else:
                    segment_colors.append(grey)
            # Gradient for red-to-grey or grey-to-red
            elif rank_range >= 10 and (e1 >= 0.05 or e2 >= 0.05):
                # Only apply gradient if one of the points is red and rank_range >= 10
                if e1 >= 0.05 and e2 < 0.05:
                    c0 = to_rgba(red)
                    c1 = to_rgba(very_light_grey)
                elif e1 < 0.05 and e2 >= 0.05:
                    c0 = to_rgba(very_light_grey)
                    c1 = to_rgba(red)
                else:
                    c0 = to_rgba(grey)
                    c1 = to_rgba(grey)
                cmap = LinearSegmentedColormap.from_list('custom_gradient', [c0, c1])
                n_steps = 100
                grad_colors = [cmap(j / (n_steps - 1)) for j in range(n_steps)]
                seg_x = np.linspace(segments[i][0][0], segments[i][1][0], n_steps)
                seg_y = np.linspace(segments[i][0][1], segments[i][1][1], n_steps)
                for j in range(n_steps - 1):
                    ax.plot([seg_x[j], seg_x[j+1]], [seg_y[j], seg_y[j+1]], color=grad_colors[j], linewidth=2, zorder=2)
                continue  # Skip adding to LineCollection for this segment
            # Gradient for grey to very light grey or vice versa (not red)
            elif (e1 >= 0.05 and e2 < 0.05) or (e1 < 0.05 and e2 >= 0.05):
                c0 = to_rgba(grey) if e1 >= 0.05 else to_rgba(very_light_grey)
                c1 = to_rgba(very_light_grey) if e1 >= 0.05 else to_rgba(grey)
                cmap = LinearSegmentedColormap.from_list('custom_gradient_grey', [c0, c1])
                n_steps = 100
                grad_colors = [cmap(j / (n_steps - 1)) for j in range(n_steps)]
                seg_x = np.linspace(segments[i][0][0], segments[i][1][0], n_steps)
                seg_y = np.linspace(segments[i][0][1], segments[i][1][1], n_steps)
                for j in range(n_steps - 1):
                    ax.plot([seg_x[j], seg_x[j+1]], [seg_y[j], seg_y[j+1]], color=grad_colors[j], linewidth=2, zorder=2)
                continue  # Skip adding to LineCollection for this segment
            else:
                # Default to high importance grey if not caught above
                segment_colors.append(grey)
        # For non-gradient segments, add to LineCollection
            lc = LineCollection([segments[i]], colors=[segment_colors[-1]], linewidths=2)
            ax.add_collection(lc)

        # Plot points individually
        for i, (xi, yi, ei) in enumerate(zip(x, y, effects)):
            if ei >= 0.05 and rank_range >= 10:
                color = red
            elif ei < 0.05:
                color = very_light_grey
            else:
                color = grey
            ax.scatter(xi, yi, color=color, zorder=3)

        # Add label to the left of the first point
        ax.text(
            -0.1,
            y[0],
            f"{param}",
            fontsize=fontsize - 3,
            va='center',
            ha='right'
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.invert_yaxis()
    ax.axes.yaxis.set_visible(False)

    # plt.xlabel("Mesh", fontsize=fontsize)

    if title is None:
        title = f"Bump Chart of Parameter Rankings in Different Meshes"
    # title = format_title_with_bold_latex(title)
    plt.title(title, fontsize=fontsize + 2, fontweight='bold')
    plt.xticks(ticks=np.arange(len(all_scenarios)), labels=legend, fontsize=fontsize, rotation=30)
    plt.tight_layout()

    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, figname)
    plt.savefig(output_path, dpi=300)
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
    parser.add_argument('--ylabels_dict', type=str, required=True, help='Path to the ylabels dictionary file (optional)')
    parser.add_argument('--xlabels_dict', type=str, required=True, help='Path to the xlabels dictionary file (optional)')
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
        ylabels_dict=args.ylabels_dict,
        xlabels_dict=args.xlabels_dict,
        plot_radar_chart=args.plot_radar_chart
    )

if __name__ == "__main__":
    main()