import argparse
import datetime
from fpdf import FPDF
import GPErks_modified.gp.data.dataset
import GPErks_modified.gp.data.dataset
import GPErks_modified.gp.experiment
import GPErks_modified.perks.cross_validation
import GPErks_modified.train.early_stop
import GPErks_modified.utils.metrics
import GPErks_modified.utils.random
from GPErks_modified.log.logger import get_logger
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.model_selection
import sys
import torch
import torchmetrics
from tqdm import tqdm

log = get_logger()

def train_gpe_kfcv(seed, emulators_folder_base, X_, y_all, x_labels, y_labels, feature_idx, metrics, mask):
    y_feature = y_all[:,feature_idx]

    if len(mask) <= len(y_feature):
        y_ = y_feature[mask]
    else:
        y_ = y_feature

    emulators_folder = f"{emulators_folder_base}/{y_labels[feature_idx]}"

    # split dataset in training and validation sets
    X, X_test, y, y_test = sklearn.model_selection.train_test_split(
        X_,
        y_,
        test_size=0.2,
        random_state=seed
    )

    dataset = GPErks_modified.gp.data.dataset.Dataset(
    X,
    y,
    x_labels=x_labels,
    y_label=y_labels[feature_idx]
)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean_function = gpytorch.means.LinearMean(input_size=dataset.input_size)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dataset.input_size))

    experiment = GPErks_modified.gp.experiment.GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        seed=seed,
        learn_noise=True
    )

    device = "cpu"
    devices = [device]
    kfcv = GPErks_modified.perks.cross_validation.KFoldCrossValidation(experiment, devices, n_splits=5, max_workers=1)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    esc = GPErks_modified.train.early_stop.GLEarlyStoppingCriterion(
        max_epochs=1000, alpha=0.1, patience=8
    )

    best_model_dct, best_train_stats_dct, test_scores_dct = kfcv.train(
        optimizer,
        esc,
        leftout_is_val=True
    )

    best_epochs = []
    for i, bts in best_train_stats_dct.items():
        best_epochs.append( bts.best_epoch )

    os.makedirs(emulators_folder, exist_ok=True)

    with open(f"{emulators_folder}/training_summary.txt", "w") as f:
        sys_out = sys.stdout
        sys.stdout = f
        
        # Format Best Epoch
        print("Best epoch\n")
        for i, epoch in enumerate(best_epochs):
            print(f"Split {i}: {epoch}")
        
        print("\nTest Scores Dictionary\n")
        
        # Extract metric names and values
        metrics = list(test_scores_dct.keys())
        scores_per_split = zip(*test_scores_dct.values())
        
        # Print metrics header
        print(" | ".join(metrics) + "\n")
        
        # Print each split's scores
        for i, scores in enumerate(scores_per_split):
            formatted_scores = " | ".join(f"{score:.6f}" for score in scores)
            print(f"Split {i}: {formatted_scores}")
        
        sys.stdout = sys_out

    np.savetxt(f"{emulators_folder}/X_train.txt",X,fmt="%g")
    np.savetxt(f"{emulators_folder}/X_test.txt",X_test,fmt="%g")
    np.savetxt(f"{emulators_folder}/y_train.txt",y,fmt="%g")
    np.savetxt(f"{emulators_folder}/y_test.txt",y_test,fmt="%g")

    max_epochs = int( np.mean(best_epochs) )  # making use of cross-validation knowledge

    return emulators_folder, X, y, X_test, y_test, max_epochs


def train_gpe_whole_dataset(seed, basefolder, emulators_folder, X, y, X_test, y_test, x_labels, y_labels, feature_idx, metrics, max_epochs, emulators_folder_name, X_all):
    dataset = GPErks_modified.gp.data.dataset.Dataset(
    X,
    y,
    X_test=X_test,
    y_test=y_test,
    x_labels=x_labels,
    y_label=y_labels[feature_idx]
    )  

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean_function = gpytorch.means.LinearMean(input_size=dataset.input_size)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dataset.input_size))

    experiment = GPErks_modified.gp.experiment.GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        seed=seed,  # reproducible training
        learn_noise=True
    )
    device = "cpu"

    emulator = GPErks_modified.train.emulator.GPEmulator(experiment, device)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)

    esc = GPErks_modified.train.early_stop.NoEarlyStoppingCriterion(max_epochs)

    os.makedirs(emulators_folder, exist_ok=True)

    snpc = GPErks_modified.train.snapshot.NeverSaveSnapshottingCriterion(
            GPErks_modified.serialization.path.posix_path(
                f"{emulators_folder}/",
                GPErks_modified.constants.DEFAULT_TRAIN_SNAPSHOT_RESTART_TEMPLATE,
            ),
            GPErks_modified.constants.DEFAULT_TRAIN_SNAPSHOT_EPOCH_TEMPLATE,
        )

    best_model, best_train_stats = emulator.train(
        optimizer,
        esc,
        snapshotting_criterion=snpc
    )

    experiment.save_to_config_file(f"{emulators_folder}/emulator.ini")

    inference = GPErks_modified.perks.inference.Inference(emulator)

    with open(f"{emulators_folder}/training_summary.txt", "a") as f:
        sys_out = sys.stdout
        sys.stdout = f
        print("\n*** Final GPE ***")
        inference.summary()
        sys.stdout = sys_out
    os.makedirs(f"{basefolder}/figures/gpe_inference/{emulators_folder_name}", exist_ok=True)

    plot_inference(inference=inference, savepath=f"{basefolder}/figures/gpe_inference/{emulators_folder_name}", figname=f"gpe_inference_{y_labels[feature_idx]}")


    flag_suspicious_simulations(inference = inference, 
                                biomarker = y_labels[feature_idx], 
                                biomarker_index = feature_idx, 
                                savepath = f"{basefolder}/figures/gpe_inference/{emulators_folder_name}", 
                                filename = "problematic_simulations.txt",
                                y_test = y_test,
                                X_test = X_test,
                                X=X_all)

def plot_inference(inference, savepath, figname):
    fig, axis = plt.subplots(1, 1, figsize=(2 * GPErks_modified.constants.WIDTH, 2 * GPErks_modified.constants.HEIGHT / 3))

    idx_sort = np.argsort(
        inference.y_pred_mean
    )  # let's sort predicted values for a better visualisation
    x = np.arange(len(idx_sort))

    ci = 1.96  # 95% confidence interval

    axis.scatter(
        x,
        inference.y_test[idx_sort],
        facecolors="none",
        edgecolors="C0",
        label="observed",
    )
    axis.scatter(
        x,
        inference.y_pred_mean[idx_sort],
        facecolors="C0",
        s=16,
        label="predicted",
    )
    axis.errorbar(
        x,
        inference.y_pred_mean[idx_sort],
        yerr=ci * inference.y_pred_std[idx_sort],
        c="C0",
        ls="none",
        lw=0.5,
        label=f"uncertainty ({ci} STD)",
    )

    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.legend(loc="upper left")

    fig.tight_layout()
    plt.savefig(f"{savepath}/{figname}.png")
    plt.close()

def flag_suspicious_simulations(inference, biomarker, biomarker_index, savepath, filename, y_test, X_test, X):


    idx_sort = np.argsort(
        inference.y_pred_mean
    ) 


    simulations_means = inference.y_test

    emulator_means = inference.y_pred_mean

    ci = 1.96  # 95% confidence interval

    emulator_uncertainty = ci * inference.y_pred_std

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    low_values = []
    high_values =[]

    low_value = False


    for problematic_index in range(len(simulations_means)):
        problematic = False
        if simulations_means[problematic_index] < emulator_means[problematic_index] - emulator_uncertainty[problematic_index]:
            problematic = True
            low_value = True
        elif simulations_means[problematic_index] > emulator_means[problematic_index] + emulator_uncertainty[problematic_index]:
            problematic = True
            low_value = False

        if problematic:
            problematic_sim_test = np.where(y_test == simulations_means[problematic_index])[0]
            if len(problematic_sim_test) > 1:
                problematic_sim_test = problematic_sim_test[0]

            param_problematic_sim = X_test[problematic_sim_test, :]

            problematic_sim = np.where(np.all(X == param_problematic_sim, axis=1))[0]
            
            if len(problematic_sim) > 1:
                problematic_sim = problematic_sim[0]

            if low_value:
                low_values.append(problematic_sim)
            else:
                high_values.append(problematic_sim)

    flattened_low_values = [item for sublist in low_values for item in np.ravel(sublist)]
    flattened_high_values = [item for sublist in high_values for item in np.ravel(sublist)]

    string_to_write = ""
    if len(flattened_low_values) > 1:
        string_to_write += f"{timestamp} - Check cycles {', '.join(map(str, flattened_low_values))} for a low value of {biomarker}.\n"
    elif len(flattened_low_values) == 1:
        string_to_write += f"{timestamp} - Check cycle {flattened_low_values[0]} for a low value of {biomarker}.\n"
    if len(flattened_high_values) > 1:
        string_to_write += f"{timestamp} - Check cycles {', '.join(map(str, flattened_high_values))} for a high value of {biomarker}.\n"
    elif len(flattened_high_values) == 1:
        string_to_write += f"{timestamp} - Check cycle {flattened_high_values[0]} for a high value of {biomarker}.\n"

    with open(f"{savepath}/{filename}", "a") as file:
        file.write(string_to_write)


def create_pdf_with_table(ylabels_path, emulators_folder, output_pdf_path, n_train_set, bold_labels, strocchi_labels):
    # Initialize PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", style="B", size=12)
    title = f"Trained on {int(n_train_set)} simulations"
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)  # Add some space after the title

    # Read ylabels
    with open(ylabels_path, "r") as f:
        ylabels = [line.strip() for line in f.readlines()]

    # Create header row
    header_row = ["Output"]
    first_output_folder = os.path.join(emulators_folder, ylabels[0])
    first_summary_file = os.path.join(first_output_folder, "training_summary.txt")

    # Parse the first file to determine additional columns
    with open(first_summary_file, "r") as f:
        lines = f.readlines()
        start_index = lines.index("*** Final GPE ***\n") + 2
        columns = [line.split()[0] for line in lines[start_index:]]
    header_row.extend(columns)

    # Add header row to PDF
    cell_width = 40  # Adjust as needed
    pdf.set_font("Arial", style="B", size=10)
    for header in header_row:
        pdf.cell(cell_width, 10, header[:12] + "..." if len(header) > 12 else header, border=1, align='C')
    pdf.ln()

    # Parse each output
    pdf.set_font("Arial", size=10)
    for ylabel in ylabels:
        output_folder = os.path.join(emulators_folder, ylabel)
        summary_file = os.path.join(output_folder, "training_summary.txt")

        if not os.path.exists(summary_file):
            raise FileNotFoundError(f"Training summary file not found for {ylabel}.")

        with open(summary_file, "r") as f:
            lines = f.readlines()

            # Parse the final GPE section
            start_index = lines.index("*** Final GPE ***\n") + 2
            gpe_scores = {}
            for line in lines[start_index:]:
                parts = line.split()
                metric_name = parts[0]
                score = float(parts[-1])  # The last part is the score
                gpe_scores[metric_name] = score

            # Determine row background color based on R2Score
            r2_score = gpe_scores.get("R2Score", None)
            if r2_score is not None:
                r2_score = float(r2_score)
                if r2_score > 0.9:
                    pdf.set_fill_color(204, 255, 204)  # Pastel green
                elif 0.7 <= r2_score <= 0.9:
                    pdf.set_fill_color(255, 255, 204)  # Yellow
                else:
                    pdf.set_fill_color(255, 204, 204)  # Red
            else:
                pdf.set_fill_color(255, 255, 255)  # Default white

            # Create row for this output
            row = [ylabel]
            for column in columns:
                row.append(gpe_scores.get(column, "N/A"))

       # Add row to PDF
        for idx, cell in enumerate(row):
            if idx == 0:
                if cell in bold_labels and cell in strocchi_labels:
                    pdf.set_font("Arial", style="BU", size=10)  # Bold and Underlined
                elif cell in bold_labels:
                    pdf.set_font("Arial", style="B", size=10)  # Bold
                elif cell in strocchi_labels:
                    pdf.set_font("Arial", style="U", size=10)  # Underlined
                else:
                    pdf.set_font("Arial", size=10)
            else:
                pdf.set_font("Arial", size=10)
            pdf.cell(cell_width, 10, str(cell), border=1, align='C', fill=True)
        pdf.ln()

    # Output the PDF
    pdf.output(output_pdf_path)

def create_pdf_with_table_second_approach(ylabels_path, emulators_folder, output_pdf_path, n_train_set, bold_labels, strocchi_labels):
    # Initialize PDF
    pdf = FPDF(orientation='L', unit='mm', format='A3')  # Landscape orientation
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", style="B", size=12)
    title = f"Trained on {int(n_train_set)} simulations"
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)  # Add some space after the title

    # Read ylabels
    with open(ylabels_path, "r") as f:
        ylabels = [line.strip() for line in f.readlines()]

    # Create header row
    header_row = ["Output"]
    first_output_folder = os.path.join(emulators_folder, ylabels[0])
    first_summary_file = os.path.join(first_output_folder, "training_summary.txt")

    # Parse the first file to determine additional columns
    with open(first_summary_file, "r") as f:
        lines = f.readlines()
        start_index = lines.index("Test Scores Dictionary\n") + 2
        columns = lines[start_index].strip().split(" | ")
    header_row.extend(columns)

    # Calculate cell width based on the number of columns
    cell_width = 80

    # Add header row to PDF
    pdf.set_font("Arial", style="B", size=10)
    for header in header_row:
        pdf.cell(cell_width, 10, header, border=1, align='C')
    pdf.ln()

    # Parse each output
    pdf.set_font("Arial", size=10)
    for ylabel in ylabels:
        output_folder = os.path.join(emulators_folder, ylabel)
        summary_file = os.path.join(output_folder, "training_summary.txt")

        with open(summary_file, "r") as f:
            lines = f.readlines()
            start_index = lines.index("Median scores\n") + 1
            scores = lines[start_index].strip().split(" | ")

        row = [ylabel] + scores

        # Add row to PDF
        for idx, cell in enumerate(row):
            if idx == 0:
                pdf.set_font("Arial", style="B", size=10)  # Bold for the first column
            else:
                pdf.set_font("Arial", size=10)
            pdf.cell(cell_width, 10, str(cell), border=1, align='C')
        pdf.ln()

    # Output the PDF
    pdf.output(output_pdf_path)


def train_gpe_kfcv_second_approach(seed, emulators_folder_base, X_, y_all, x_labels, y_labels, feature_idx, metrics, mask, device, n_folds):
    y_feature = y_all[:,feature_idx]

    if len(mask) <= len(y_feature):
        y_ = y_feature[mask]
    else:
        y_ = y_feature

    emulators_folder = f"{emulators_folder_base}/{y_labels[feature_idx]}"

    # split dataset in training and validation sets
    # X, X_test, y, y_test = sklearn.model_selection.train_test_split(
    #     X_,
    #     y_,
    #     test_size=0.2,
    #     random_state=seed
    # )

    X = X_
    y = y_

    dataset = GPErks_modified.gp.data.dataset.Dataset(
    X,
    y,
    x_labels=x_labels,
    y_label=y_labels[feature_idx]
)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean_function = gpytorch.means.LinearMean(input_size=dataset.input_size)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dataset.input_size))

    experiment = GPErks_modified.gp.experiment.GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        seed=seed,
        learn_noise=True
    )

    devices = [device]
    kfcv = GPErks_modified.perks.cross_validation.KFoldCrossValidation(experiment, devices, n_splits=n_folds, max_workers=1)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
    esc = GPErks_modified.train.early_stop.GLEarlyStoppingCriterion(
        max_epochs=1000, alpha=0.1, patience=8
    )

    best_model_dct, best_train_stats_dct, test_scores_dct = kfcv.train(
        optimizer,
        esc,
        leftout_is_val=True
    )

    best_epochs = []
    for i, bts in best_train_stats_dct.items():
        best_epochs.append( bts.best_epoch )

    os.makedirs(emulators_folder, exist_ok=True)

    with open(f"{emulators_folder}/training_summary.txt", "w") as f:
        sys_out = sys.stdout
        sys.stdout = f
        
        # Format Best Epoch
        print("Best epoch\n")
        for i, epoch in enumerate(best_epochs):
            print(f"Split {i}: {epoch}")
        
        print("\nTest Scores Dictionary\n")
        
        # Extract metric names and values
        metrics = list(test_scores_dct.keys())
        scores_per_split = list(zip(*test_scores_dct.values()))
        
        # Print metrics header
        print(" | ".join(metrics) + "\n")
        
        # Print each split's scores
        for i, scores in enumerate(scores_per_split):
            formatted_scores = " | ".join(f"{score:.6f}" for score in scores)
            print(f"Split {i}: {formatted_scores}")

        # Print the median of the test scores

        median_score = np.median(scores_per_split, axis=0)

        print(f"\nMedian scores\n {' | '.join([f'{score:.6f}' for score in median_score])}")
        
        sys.stdout = sys_out


    max_epochs = int( np.mean(best_epochs) )  # making use of cross-validation knowledge

    return emulators_folder


def train_gpe_whole_dataset_second_approach(seed, emulators_folder, X, y, x_labels, y_labels, feature_idx, metrics, device):

    dataset = GPErks_modified.gp.data.dataset.Dataset(
    X,
    y[:,feature_idx],
    x_labels=x_labels,
    y_label=y_labels[feature_idx]
    )  

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean_function = gpytorch.means.LinearMean(input_size=dataset.input_size)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dataset.input_size))

    experiment = GPErks_modified.gp.experiment.GPExperiment(
        dataset,
        likelihood,
        mean_function,
        kernel,
        n_restarts=3,
        metrics=metrics,
        seed=seed,  # reproducible training
        learn_noise=True
    )

    emulator = GPErks_modified.train.emulator.GPEmulator(experiment, device)

    optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)

    esc = GPErks_modified.train.early_stop.NoEarlyStoppingCriterion(
        max_epochs=1000
    )

    os.makedirs(emulators_folder, exist_ok=True)

    snpc = GPErks_modified.train.snapshot.NeverSaveSnapshottingCriterion(
            GPErks_modified.serialization.path.posix_path(
                f"{emulators_folder}/",
                GPErks_modified.constants.DEFAULT_TRAIN_SNAPSHOT_RESTART_TEMPLATE,
            ),
            GPErks_modified.constants.DEFAULT_TRAIN_SNAPSHOT_EPOCH_TEMPLATE,
        )

    best_model, best_train_stats = emulator.train(
        optimizer,
        esc,
        snapshotting_criterion=snpc
    )

    experiment.save_to_config_file(f"{emulators_folder}/emulator.ini")


def main_first_approach(args):

    basefolder = args.basefolder
    feature_idx = int(args.feature_idx)
    output_mask_name = args.output_mask_name
    n_train = args.n_train
    emulators_folder_name = args.emulators_folder_name

    seed = 8
    GPErks_modified.utils.random.set_seed(seed)

    emulators_folder_base = F"{basefolder}/output/{emulators_folder_name}/"
    X_all =  np.loadtxt(f"{basefolder}/data/X.txt", dtype=float)
    mask =  np.loadtxt(f"{basefolder}/output/{output_mask_name}", dtype=float)
    mask = mask.astype(bool)
    input_masked = X_all[:mask.shape[0]]  # Trim X_all to match the size of the mask
    X_ = input_masked[mask]
    Y_original = np.loadtxt(f"{basefolder}/data/Y.txt", dtype=float)

    if n_train > 0:
        final_n_train = min(int(n_train/0.8), len(X_))
    else:
        final_n_train = len(X_)

    X_ = X_[:final_n_train]
    y_all = Y_original[:final_n_train]

    with open(f"{basefolder}/data/xlabels.txt", "r") as f:
        x_labels = f.read().splitlines()

    with open(f"{basefolder}/data/ylabels.txt", "r") as f:
            y_labels = f.read().splitlines()

    metrics = [GPErks_modified.utils.metrics.IndependentStandardErrorMetric(), torchmetrics.MeanSquaredError(), torchmetrics.R2Score()]

    if feature_idx == -1:
        feature_array = [i for i in range(len(y_labels))]
    else:
        feature_array = [feature_idx]

    

    # Create Y by selecting the corresponding rows from Y_original
    Y_dense = []
    Y_index = 0

    for i in range(len(mask)):
        if mask[i] == 1:
            Y_dense.append(Y_original[Y_index])
            Y_index += 1
        else:
            Y_dense.append(np.zeros(Y_original.shape[1]))

    # Convert Y to a NumPy array
    Y_dense = np.array(Y_dense)

    # Add tqdm progress bar
    t = tqdm(feature_array, colour="blue")
    for feature_idx2 in t:
        
        if not os.path.isfile(f"{basefolder}/figures/gpe_inference/{emulators_folder_name}/gpe_inference_{y_labels[feature_idx2]}.png"):
            t.set_description(f"Training feature: {y_labels[feature_idx2]}")

            emulators_folder, X, y, X_test, y_test, max_epochs = train_gpe_kfcv(seed=seed,
                        mask=mask,
                        emulators_folder_base=emulators_folder_base, 
                        X_=X_, 
                        y_all=y_all, 
                        x_labels=x_labels, 
                        y_labels=y_labels, 
                        feature_idx=feature_idx2, 
                        metrics=metrics)

            train_gpe_whole_dataset(seed=seed, 
                                    basefolder=basefolder, 
                                    emulators_folder=emulators_folder, 
                                    X=X, 
                                    y=y, 
                                    X_test=X_test, 
                                    y_test=y_test, 
                                    x_labels=x_labels, 
                                    y_labels=y_labels, 
                                    feature_idx=feature_idx2, 
                                    metrics=metrics, 
                                    max_epochs=max_epochs,
                                    emulators_folder_name = emulators_folder_name,
                                    X_all=X_all)
        
    
    n_train_set = 0.8*np.shape(X_)[0]
    create_pdf_with_table(ylabels_path=f"{basefolder}/data/ylabels.txt", emulators_folder=emulators_folder_base, output_pdf_path=f"{emulators_folder_base}/summary_metrics.pdf", n_train_set=n_train_set, bold_labels=["LVedv","LVedp","LVesv","LVpmax","LVEF","A_TAT","V_TAT"],
    strocchi_labels =  ["LVedv","LVedp","LVesv","LVpmax","LVdpdtMax","LVdpdtMin","LAedv","LAesv","LApMax","LAinflV","RVedv","RVedp","RVesv","RVpmax","RVdpdtMax","RVdpdtMin","RAedv","RAesv","RApMax","RAinflV"])

def main_second_approach(args):

    #### We do first k-fold cross-validation to obtain the median quality of the model. We then train using ALL the data and save the model.

    basefolder = args.basefolder
    feature_idx = int(args.feature_idx)
    output_mask_name = args.output_mask_name
    n_train = args.n_train
    emulators_folder_name = args.emulators_folder_name

    seed = 8
    GPErks_modified.utils.random.set_seed(seed)

    emulators_folder_base = F"{basefolder}/output/{emulators_folder_name}/"
    X_all =  np.loadtxt(f"{basefolder}/data/X.txt", dtype=float)
    mask =  np.loadtxt(f"{basefolder}/output/{output_mask_name}", dtype=float)
    mask = mask.astype(bool)
    input_masked = X_all[:mask.shape[0]]  # Trim X_all to match the size of the mask
    X_ = input_masked[mask]
    Y_original = np.loadtxt(f"{basefolder}/data/Y.txt", dtype=float)

    if n_train > 0:
        final_n_train = min(int(n_train), len(X_))
    else:
        final_n_train = len(X_)

    if args.n_folds > 0:
        n_folds = args.n_folds
    else:
        n_folds = final_n_train + args.n_folds + 1

    log.warning(f"Splits: {n_folds}")

    X_ = X_[:final_n_train]
    y_all = Y_original[:final_n_train]

    with open(f"{basefolder}/data/xlabels.txt", "r") as f:
        x_labels = f.read().splitlines()

    with open(f"{basefolder}/data/ylabels.txt", "r") as f:
            y_labels = f.read().splitlines()

    metrics = [GPErks_modified.utils.metrics.IndependentStandardErrorMetric(),
                torchmetrics.MeanAbsolutePercentageError(), 
                torchmetrics.SymmetricMeanAbsolutePercentageError(),
                torchmetrics.MeanSquaredLogError(),
                torchmetrics.MeanSquaredError()]

    if feature_idx == -1:
        feature_array = [i for i in range(len(y_labels))]
    else:
        feature_array = [feature_idx]

    

    # Create Y by selecting the corresponding rows from Y_original
    Y_dense = []
    Y_index = 0

    for i in range(len(mask)):
        if mask[i] == 1:
            Y_dense.append(Y_original[Y_index])
            Y_index += 1
        else:
            Y_dense.append(np.zeros(Y_original.shape[1]))

    # Convert Y to a NumPy array
    Y_dense = np.array(Y_dense)

    # Add tqdm progress bar
    t = tqdm(feature_array, colour="blue")
    for feature_idx2 in t:
        
        if not os.path.isfile(f"{basefolder}/figures/gpe_inference/{emulators_folder_name}/gpe_inference_{y_labels[feature_idx2]}.png"):
            t.set_description(f"Training feature: {y_labels[feature_idx2]}")

            emulators_folder = train_gpe_kfcv_second_approach(seed=seed,
                        mask=mask,
                        emulators_folder_base=emulators_folder_base, 
                        X_=X_, 
                        y_all=y_all, 
                        x_labels=x_labels, 
                        y_labels=y_labels, 
                        feature_idx=feature_idx2, 
                        metrics=metrics,
                        device=args.device,
                        n_folds=n_folds)

            train_gpe_whole_dataset_second_approach(seed=seed, 
                                    emulators_folder=emulators_folder, 
                                    X=X_, 
                                    y=y_all,
                                    x_labels=x_labels, 
                                    y_labels=y_labels, 
                                    feature_idx=feature_idx2, 
                                    metrics=metrics, 
                                    device=args.device)
        

    n_train_set = np.shape(X_)[0]
    create_pdf_with_table_second_approach(ylabels_path=f"{basefolder}/data/ylabels.txt", emulators_folder=emulators_folder_base, output_pdf_path=f"{emulators_folder_base}/summary_metrics.pdf", n_train_set=n_train_set, bold_labels=["LVedv","LVedp","LVesv","LVpmax","LVEF","A_TAT","V_TAT"],
    strocchi_labels =  ["LVedv","LVedp","LVesv","LVpmax","LVdpdtMax","LVdpdtMin","LAedv","LAesv","LApMax","LAinflV","RVedv","RVedp","RVesv","RVpmax","RVdpdtMax","RVdpdtMin","RAedv","RAesv","RApMax","RAinflV"])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        default="/media/croderog/SeagateExpansionDrive/HCM/2/scenarios/41",
                        help='Path to the folder where the simulations, data, and figure folders are.')
    parser.add_argument('--feature_idx', default=-1)
    parser.add_argument('--output_mask_name', default="output_mask_beat_5.txt")
    parser.add_argument('--n_train', default=-1, type=int)
    parser.add_argument('--n_folds', default=5, type=int)
    parser.add_argument('--emulators_folder_name', default="emulators")
    parser.add_argument('--device', choices=["cpu", "cuda"], default="cpu")

    args = parser.parse_args()

    # main_first_approach(args)
    main_second_approach(args)
