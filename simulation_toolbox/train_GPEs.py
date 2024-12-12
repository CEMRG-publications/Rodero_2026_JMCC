import argparse
import GPErks_modified.gp.data.dataset
import GPErks_modified.gp.data.dataset
import GPErks_modified.gp.experiment
import GPErks_modified.perks.cross_validation
import GPErks_modified.train.early_stop
import GPErks_modified.utils.metrics
import GPErks_modified.utils.random
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.model_selection
import sys
import torch
import torchmetrics

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



def train_gpe_whole_dataset(seed, basefolder, emulators_folder, X, y, X_test, y_test, x_labels, y_labels, feature_idx, metrics, max_epochs):
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

    plot_inference(inference=inference, savepath=f"{basefolder}/figures", figname=f"gpe_inference_{y_labels[feature_idx]}")

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

import os
from fpdf import FPDF

def create_pdf_with_table(ylabels_path, emulators_folder, output_pdf_path):
    # Initialize PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=10)

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
        for cell in row:
            pdf.cell(cell_width, 10, str(cell), border=1, align='C', fill=True)
        pdf.ln()

    # Output the PDF
    pdf.output(output_pdf_path)




def main(args):

    basefolder = args.basefolder
    feature_idx = int(args.feature_idx)
    output_mask_name = args.output_mask_name

    seed = 8
    GPErks_modified.utils.random.set_seed(seed)

    emulators_folder_base = F"{basefolder}/output/emulators/"
    X_all =  np.loadtxt(f"{basefolder}/data/X.txt", dtype=float)
    mask =  np.loadtxt(f"{basefolder}/output/{output_mask_name}", dtype=float)
    mask = mask.astype(bool)
    input_masked = X_all[:mask.shape[0]]  # Trim X_all to match the size of the mask
    X_ = input_masked[mask]
    y_all = np.loadtxt(f"{basefolder}/data/Y.txt", dtype=float)

    with open(f"{basefolder}/data/xlabels.txt", "r") as f:
        x_labels = f.read().splitlines()

    with open(f"{basefolder}/data/ylabels.txt", "r") as f:
            y_labels = f.read().splitlines()

    metrics = [GPErks_modified.utils.metrics.IndependentStandardErrorMetric(), torchmetrics.MeanSquaredError(), torchmetrics.R2Score()]

    if feature_idx == -1:
        feature_array = [i for i in range(len(y_labels))]
    else:
        feature_array = [feature_idx]

    for feature_idx2 in feature_array:

        if not os.path.isfile(f"{basefolder}/figures/gpe_inference_{y_labels[feature_idx2]}.png"):

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
                                    max_epochs=max_epochs)
        
    
    output_pdf_path = f"{emulators_folder_base}/output_table.pdf"
    create_pdf_with_table(ylabels_path=f"{basefolder}/data/ylabels.txt", emulators_folder=emulators_folder_base, output_pdf_path=f"{emulators_folder_base}/summary_metrics.pdf")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        default="/media/croderog/SeagateExpansionDrive/HCM/2/scenarios/41",
                        help='Path to the folder where the simulations, data, and figure folders are.')
    parser.add_argument('--feature_idx', default=-1)
    parser.add_argument('--output_mask_name', default="output_mask_beat_5.txt")

    args = parser.parse_args()

    main(args)
