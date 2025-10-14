import argparse
import json
import os
import re
import numpy as np
from fpdf import FPDF


def extract_n_folds(emulators_folder_name):
    """Extract number of folds from folder name like 'emulators_5fold'"""
    match = re.search(r'(\d+)fold', emulators_folder_name)
    return int(match.group(1)) if match else None


def format_number(value, decimals=2):
    """Format number with specified decimals or scientific notation if too small"""
    if abs(value) < 10 ** (-decimals) and value != 0:
        return f"{value:.2e}"
    else:
        return f"{value:.{decimals}f}"


def get_metric_display_name(metric):
    """Convert metric code names to readable names"""
    metric_names = {
        "IndependentStandardErrorMetric": "ISE",
        "MeanAbsolutePercentageError": "MAPE",
        "SymmetricMeanAbsolutePercentageError": "SMAPE",
        "MeanSquaredLogError": "MSLE",
        "MeanSquaredError": "MSE",
        "R2Score": "R²"
    }
    return metric_names.get(metric, metric)


def create_performance_table_pdf(basefolder, emulators_folder_name, ylabels_json_path,
                                 output_pdf_path, title_preffix, trim_percentage=0):
    """Generate a landscape A4 PDF summarizing emulator performance metrics"""

    # Load which outputs to include
    with open(ylabels_json_path, 'r') as f:
        ylabels_dict = json.load(f)
    outputs_to_run = {k: v for k, v in ylabels_dict.items() if v['run'] == 1}

    n_folds = extract_n_folds(emulators_folder_name) or "Unknown"
    emulators_folder = f"{basefolder}/output/{emulators_folder_name}"

    # Get number of splits and metrics
    first_output = list(outputs_to_run.keys())[0]
    first_summary = f"{emulators_folder}/{first_output}/training_summary.txt"
    with open(first_summary, 'r') as f:
        lines = f.readlines()
        n_splits = sum(1 for line in lines if line.startswith("Split "))
        start_index = lines.index("Test Scores Dictionary\n") + 2
        metrics = lines[start_index].strip().split(" | ")

    # Total number of simulations
    Y_data = np.loadtxt(f"{basefolder}/data/Y.txt")
    n_total_sims = Y_data.shape[0]

    # Metric bin widths
    bin_widths = {
        "IndependentStandardErrorMetric": 1.0,
        "MeanAbsolutePercentageError": 1e-3,
        "SymmetricMeanAbsolutePercentageError": 1e-3,
        "MeanSquaredLogError": 1e-3,
        "MeanSquaredError": 1,
        "R2Score": 5e-2
    }

    # ✅ Initialize PDF (A4 landscape)
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()

    # ✅ Add full DejaVu font set (handles UTF-8)
    pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", uni=True)
    pdf.add_font("DejaVu", "I", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", uni=True)
    pdf.add_font("DejaVu", "BI", "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf", uni=True)

    # Title
    pdf.set_font("DejaVu", "B", size=14)
    title = f"{title_preffix} - {n_folds}-fold cross validation trained on {n_total_sims} simulations"
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(5)

    # Layout constants
    n_metrics = len(metrics)
    page_width = 297  # A4 landscape width
    margin = 15
    available_width = page_width - 2 * margin
    label_width = 40
    metric_total_width = (available_width - label_width) / n_metrics
    stat_width = metric_total_width / 3

    # Header row
    pdf.set_font("DejaVu", "B", size=9)
    pdf.cell(label_width, 8, "Output", border=1, align='C')
    for metric in metrics:
        pdf.cell(metric_total_width, 8, get_metric_display_name(metric), border=1, align='C')
    pdf.ln()

    # Sub-header
    pdf.set_font("DejaVu", "I", size=7)
    pdf.cell(label_width, 6, "", border=1, align='C')
    for _ in metrics:
        pdf.cell(stat_width, 6, "Mean", border=1, align='C')
        pdf.cell(stat_width, 6, "Median", border=1, align='C')
        pdf.cell(stat_width, 6, "Mode", border=1, align='C')
    pdf.ln()

    # Table body
    pdf.set_font("DejaVu", size=7)

    for output_name, output_info in outputs_to_run.items():
        summary_file = f"{emulators_folder}/{output_name}/training_summary.txt"
        if not os.path.exists(summary_file):
            continue

        # Read scores
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        scores_dict = {metric: [] for metric in metrics}
        for line in lines:
            if line.startswith("Split "):
                scores_raw = line.split(":")[1].strip().split(" | ")
                for metric, score in zip(metrics, scores_raw):
                    scores_dict[metric].append(float(score))

        # Optionally trim worst R²
        if trim_percentage > 0:
            r2_scores = np.array(scores_dict["R2Score"])
            num_to_trim = int(len(r2_scores) * trim_percentage / 100)
            trim_indices = np.argsort(r2_scores)[:num_to_trim]
            for metric in metrics:
                arr = np.array(scores_dict[metric])
                arr[trim_indices] = np.nan
                scores_dict[metric] = arr

        # Compute stats
        row_stats = []
        for metric in metrics:
            scores = np.array(scores_dict[metric])
            mean_val = np.nanmean(scores)
            median_val = np.nanmedian(scores)
            bin_width = bin_widths.get(metric, 1.0)
            if metric == "MeanSquaredError":
                bin_width = 0.01 * mean_val
            if metric == "R2Score" and np.nanmin(scores) < -2:
                bin_width = (np.nanmax(scores) - np.nanmin(scores)) / 100

            min_val, max_val = np.nanmin(scores), np.nanmax(scores)
            if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
                mode_val = np.nan
            else:
                bin_edges = np.arange(min_val, max_val + bin_width, bin_width)
                hist, _ = np.histogram(scores, bins=bin_edges)
                mode_bin_index = np.where(hist == np.max(hist))[0][-1]
                mode_val = bin_edges[mode_bin_index]
            row_stats.append((mean_val, median_val, mode_val))

        # Background color based on R² median
        # Gather necessary stats
        r2_median = row_stats[metrics.index("R2Score")][1]
        r2_mode = row_stats[metrics.index("R2Score")][2]
        mape_median = row_stats[metrics.index("MeanAbsolutePercentageError")][1]

        # Calculate normalized MSE (interpolated)
        mse_median = row_stats[metrics.index("MeanSquaredError")][1]
        y_mean = np.mean(Y_data)  # Assuming Y_data is loaded earlier
        mse_interp_median = np.sqrt(mse_median) / y_mean

        # Set row color based on combined rules
        if r2_median > 0.7:
            pdf.set_fill_color(204, 255, 204)  # Green
        elif r2_median < 0.7 and r2_mode > 0.7 and mape_median < 0.1 and mse_interp_median < 0.1:
            pdf.set_fill_color(255, 255, 204)  # Yellow
        else:
            pdf.set_fill_color(255, 204, 204)  # Red


        # ✅ Use plain name: underscores replaced by spaces
        display_name = output_name.replace("_", " ")

        pdf.cell(label_width, 6, display_name, border=1, align='C', fill=True)
        for mean_val, median_val, mode_val in row_stats:
            pdf.cell(stat_width, 6, format_number(mean_val), border=1, align='C', fill=True)
            pdf.cell(stat_width, 6, format_number(median_val), border=1, align='C', fill=True)
            pdf.cell(stat_width, 6, format_number(mode_val), border=1, align='C', fill=True)
        pdf.ln()

    pdf.output(output_pdf_path)
    print(f"✅ PDF saved to: {output_pdf_path}")


def print_metric_formulas():
    print("\n" + "=" * 80)
    print("METRIC FORMULAS")
    print("=" * 80 + "\n")
    print("1. ISE (Independent Standard Error Metric)")
    print("   sqrt(sum((y_true - y_pred)² / var(y_true)) / n)\n")
    print("2. MAPE (Mean Absolute Percentage Error)")
    print("   (100/n) * sum(|y_true - y_pred| / |y_true|)\n")
    print("3. SMAPE (Symmetric MAPE)")
    print("   (100/n) * sum(|y_true - y_pred| / ((|y_true| + |y_pred|)/2))\n")
    print("4. MSLE (Mean Squared Log Error)")
    print("   (1/n) * sum((log(y_true + 1) - log(y_pred + 1))²)\n")
    print("5. MSE (Mean Squared Error)")
    print("   (1/n) * sum((y_true - y_pred)²)\n")
    print("6. R² (Coefficient of Determination)")
    print("   1 - (SS_res / SS_tot)\n")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate performance table PDF from emulator results')
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--basefolder', type=str, required=True,
                        help='Path to base folder containing data and output')
    parser.add_argument('--emulators_folder_name', type=str, required=True,
                        help='Name of emulators folder (e.g., emulators_5fold)')
    parser.add_argument('--ylabels_json', type=str, required=True,
                        help='Path to JSON file with output labels')
    parser.add_argument('--output_pdf', type=str, default='performance_summary.pdf',
                        help='Output PDF filename')
    parser.add_argument('--trim_percentage', type=float, default=0,
                        help='Percentage of worst performers to trim')
    parser.add_argument('--title_preffix', type=str, default='LVOTO',
                        help='Prefix for the title in the PDF')

    args = parser.parse_args()

    print_metric_formulas()

    create_performance_table_pdf(
        basefolder=args.basefolder,
        emulators_folder_name=args.emulators_folder_name,
        ylabels_json_path=args.ylabels_json,
        output_pdf_path=args.output_pdf,
        trim_percentage=args.trim_percentage,
        title_preffix=args.title_preffix
    )
