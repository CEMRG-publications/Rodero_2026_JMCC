import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.stats import pearsonr, spearmanr
import json

from common.utils import read_xlabels_dict


class GSAComparator:
    """Compare GSA results across multiple scenarios.

    By default, runs:
      - Variance explained values
      - Parameter rankings
      - Only considers parameter-output pairs where BOTH scenarios have relevance >= 0.05
    """

    def __init__(self, scenarios, scenario_names, xlabels_file, ylabels_file,
                 xlabels_dict, ylabels_dict, gsa_mode="Si_total", mode="max", 
                 relevance_threshold=0.05):
        self.scenarios = scenarios
        self.scenario_names = scenario_names
        self.xlabels_file = xlabels_file
        self.ylabels_file = ylabels_file
        self.xlabels_dict = xlabels_dict
        self.ylabels_dict = ylabels_dict
        self.gsa_mode = gsa_mode
        self.mode = mode
        self.relevance_threshold = relevance_threshold

        # Load labels
        self.xlabels = np.loadtxt(xlabels_file, dtype=str)
        self.ylabels = np.loadtxt(ylabels_file, dtype=str)
        _, self.xlabels_dict_all = read_xlabels_dict(xlabels_dict, self.xlabels)
        
        # Load ylabels dict
        self.ylabels_dict_all = {}
        if ylabels_dict and os.path.exists(ylabels_dict):
            _, self.ylabels_dict_all = read_xlabels_dict(ylabels_dict, self.ylabels)

        # Storage for matrices (both values and rankings)
        self.gsa_matrices = {}
        self.gsa_ranking_matrices = {}
        self.all_params = None
        self.all_outputs = None

    def load_gsa_data(self):
        """Load GSA data from all scenarios into matrices."""
        print("Loading GSA data from all scenarios...")

        self.all_params = list(self.xlabels)
        self.all_outputs = list(self.ylabels)

        print(f"Using {len(self.all_params)} parameters and {len(self.all_outputs)} outputs")

        for scenario, scenario_name in zip(self.scenarios, self.scenario_names):
            matrix = np.zeros((len(self.all_params), len(self.all_outputs)))
            matrix[:] = np.nan
            ranking_matrix = np.zeros((len(self.all_params), len(self.all_outputs)))
            ranking_matrix[:] = np.nan

            output_dir = os.path.join(scenario, "output")

            for j, output_name in enumerate(self.all_outputs):
                rank_file = f"Rank_{self.gsa_mode}_{self.mode}_{output_name}.txt"
                filepath = os.path.join(output_dir, rank_file)

                if os.path.exists(filepath):
                    rank_counter = 1
                    with open(filepath, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split("\t")
                            if len(parts) >= 2:
                                param = parts[0]
                                try:
                                    value = float(parts[1])
                                except ValueError:
                                    continue

                                if param in self.all_params:
                                    i = self.all_params.index(param)
                                    matrix[i, j] = value
                                    ranking_matrix[i, j] = rank_counter
                                    rank_counter += 1

            self.gsa_matrices[scenario_name] = matrix
            self.gsa_ranking_matrices[scenario_name] = ranking_matrix
            print(f"Loaded matrix for {scenario_name}: shape {matrix.shape}")

    def flatten_matrix_with_relevance_mask(self, matrix1, matrix2):
        """Flatten matrices but only include positions where BOTH are relevant (>= threshold)."""
        flat1 = matrix1.flatten()
        flat2 = matrix2.flatten()
        
        # Create relevance mask: both must be >= threshold and not NaN
        relevant_mask = (
            (~np.isnan(flat1)) & 
            (~np.isnan(flat2)) & 
            (flat1 >= self.relevance_threshold) & 
            (flat2 >= self.relevance_threshold)
        )
        
        # Only keep relevant positions
        flat1_filtered = flat1[relevant_mask]
        flat2_filtered = flat2[relevant_mask]
        
        return flat1_filtered, flat2_filtered, np.sum(relevant_mask)

    def compute_distances(self, use_rankings=False):
        n_scenarios = len(self.scenario_names)
        matrices = self.gsa_ranking_matrices if use_rankings else self.gsa_matrices
        matrix_type = "rankings" if use_rankings else "values"

        cosine_dist = np.zeros((n_scenarios, n_scenarios))
        euclidean_dist = np.zeros((n_scenarios, n_scenarios))
        manhattan_dist = np.zeros((n_scenarios, n_scenarios))
        pearson_dist = np.zeros((n_scenarios, n_scenarios))
        spearman_dist = np.zeros((n_scenarios, n_scenarios))

        print(f"\nComputing pairwise distances using {matrix_type}...")
        print(f"Only considering parameter-output pairs where both scenarios have relevance >= {self.relevance_threshold}")

        for i in range(n_scenarios):
            for j in range(n_scenarios):
                if i == j:
                    cosine_dist[i, j] = 0
                    euclidean_dist[i, j] = 0
                    manhattan_dist[i, j] = 0
                    pearson_dist[i, j] = 0
                    spearman_dist[i, j] = 0
                else:
                    # Get value matrices to check relevance
                    values_i = self.gsa_matrices[self.scenario_names[i]]
                    values_j = self.gsa_matrices[self.scenario_names[j]]
                    
                    # Get comparison matrices (values or rankings)
                    matrix_i = matrices[self.scenario_names[i]]
                    matrix_j = matrices[self.scenario_names[j]]
                    
                    # Flatten with relevance mask based on VALUES
                    flat_i_full = matrix_i.flatten()
                    flat_j_full = matrix_j.flatten()
                    values_i_flat = values_i.flatten()
                    values_j_flat = values_j.flatten()
                    
                    # Create relevance mask based on VALUES
                    relevant_mask = (
                        (~np.isnan(values_i_flat)) & 
                        (~np.isnan(values_j_flat)) & 
                        (values_i_flat >= self.relevance_threshold) & 
                        (values_j_flat >= self.relevance_threshold)
                    )
                    
                    flat_i = flat_i_full[relevant_mask]
                    flat_j = flat_j_full[relevant_mask]
                    n_relevant = np.sum(relevant_mask)
                    
                    if n_relevant < 2:
                        print(f"Warning: Only {n_relevant} relevant pairs between {self.scenario_names[i]} and {self.scenario_names[j]}")
                        cosine_dist[i, j] = np.nan
                        euclidean_dist[i, j] = np.nan
                        manhattan_dist[i, j] = np.nan
                        pearson_dist[i, j] = np.nan
                        spearman_dist[i, j] = np.nan
                        continue

                    try:
                        cosine_dist[i, j] = cosine(flat_i, flat_j)
                    except Exception:
                        cosine_dist[i, j] = np.nan

                    euclidean_dist[i, j] = euclidean(flat_i, flat_j)
                    manhattan_dist[i, j] = cityblock(flat_i, flat_j)

                    if np.std(flat_i) > 0 and np.std(flat_j) > 0:
                        corr, _ = pearsonr(flat_i, flat_j)
                        pearson_dist[i, j] = 1 - abs(corr)
                    else:
                        pearson_dist[i, j] = np.nan

                    if np.std(flat_i) > 0 and np.std(flat_j) > 0:
                        corr, _ = spearmanr(flat_i, flat_j)
                        spearman_dist[i, j] = 1 - abs(corr)
                    else:
                        spearman_dist[i, j] = np.nan

        distance_matrices = {
            "cosine": cosine_dist,
            "euclidean": euclidean_dist,
            "manhattan": manhattan_dist,
            "pearson": pearson_dist,
            "spearman": spearman_dist,
        }

        if use_rankings:
            self.distance_matrices_rankings = distance_matrices
        else:
            self.distance_matrices = distance_matrices

        return distance_matrices

    def generate_report(self, savepath, use_rankings=False):
        suffix = "_rankings" if use_rankings else "_values"
        report_path = os.path.join(savepath, f"gsa_comparison_report{suffix}.txt")
        distance_matrices = self.distance_matrices_rankings if use_rankings else self.distance_matrices
        matrix_type = "Rankings" if use_rankings else "Variance Explained Values"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"GSA COMPARISON REPORT - {matrix_type}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Number of scenarios: {len(self.scenario_names)}\n")
            f.write(f"Scenarios: {', '.join(self.scenario_names)}\n\n")

            f.write(f"Matrix dimensions: {len(self.all_params)} parameters × {len(self.all_outputs)} outputs\n")
            f.write(f"Parameters: {len(self.all_params)}\n")
            f.write(f"Outputs: {len(self.all_outputs)}\n")
            f.write(f"Relevance threshold: {self.relevance_threshold}\n\n")

            for metric_name, dist_matrix in distance_matrices.items():
                f.write(f"\n{metric_name.upper()} DISTANCE\n")
                f.write("-" * 80 + "\n")
                df = pd.DataFrame(dist_matrix, index=self.scenario_names, columns=self.scenario_names)
                f.write(df.to_string() + "\n\n")

        print(f"Report saved to: {report_path}")

        json_path = os.path.join(savepath, f"gsa_comparison_metrics{suffix}.json")
        json_data = {
            "scenario_names": self.scenario_names, 
            "analysis_type": matrix_type, 
            "relevance_threshold": self.relevance_threshold,
            "metrics": {}
        }
        for metric_name, dist_matrix in distance_matrices.items():
            json_data["metrics"][metric_name] = dist_matrix.tolist()
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"JSON metrics saved to: {json_path}")

    def plot_distance_heatmaps(self, savepath, fontsize=10, use_rankings=False):
        suffix = "_rankings" if use_rankings else "_values"
        matrix_type = "Rankings" if use_rankings else "Values"
        distance_matrices = self.distance_matrices_rankings if use_rankings else self.distance_matrices

        print(f"\nGenerating distance heatmaps ({matrix_type})...")

        n_metrics = len(distance_matrices)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (metric_name, dist_matrix) in enumerate(distance_matrices.items()):
            ax = axes[idx]
            sns.heatmap(
                dist_matrix,
                xticklabels=self.scenario_names,
                yticklabels=self.scenario_names,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn_r",
                ax=ax,
                square=True,
                cbar_kws={"label": "Distance"},
                linewidths=0.5,
                linecolor="gray",
            )
            ax.set_title(f"{metric_name.title()} Distance ({matrix_type})", fontsize=fontsize + 2, fontweight="bold")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize - 1)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize - 1)

        if n_metrics < len(axes):
            for k in range(n_metrics, len(axes)):
                fig.delaxes(axes[k])

        plt.tight_layout()
        output_path = os.path.join(savepath, f"gsa_distance_heatmaps{suffix}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Distance heatmaps saved to: {output_path}")

    def plot_mds_visualization(self, savepath, fontsize=10, use_rankings=False):
        from sklearn.manifold import MDS

        suffix = "_rankings" if use_rankings else "_values"
        matrix_type = "Rankings" if use_rankings else "Values"
        distance_matrices = self.distance_matrices_rankings if use_rankings else self.distance_matrices

        print(f"\nGenerating MDS visualizations ({matrix_type})...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (metric_name, dist_matrix) in enumerate(distance_matrices.items()):
            ax = axes[idx]
            if np.any(np.isnan(dist_matrix)):
                ax.text(0.5, 0.5, f"NaN in {metric_name} distance", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{metric_name.title()} Distance (MDS, {matrix_type})", fontsize=fontsize + 2)
                continue

            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
            coords = mds.fit_transform(dist_matrix)

            ax.scatter(coords[:, 0], coords[:, 1], s=200, alpha=0.6)
            for i, name in enumerate(self.scenario_names):
                ax.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=fontsize - 1, ha="center", va="center")

            ax.set_title(
                f"{metric_name.title()} Distance (MDS, {matrix_type})", fontsize=fontsize + 2, fontweight="bold"
            )
            ax.set_xlabel("Dimension 1", fontsize=fontsize)
            ax.set_ylabel("Dimension 2", fontsize=fontsize)
            ax.grid(True, alpha=0.3)

        if len(distance_matrices) < len(axes):
            for k in range(len(distance_matrices), len(axes)):
                fig.delaxes(axes[k])

        plt.tight_layout()
        output_path = os.path.join(savepath, f"gsa_mds_visualization{suffix}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"MDS visualization saved to: {output_path}")

    def plot_dendrogram(self, savepath, fontsize=10, use_rankings=False):
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        suffix = "_rankings" if use_rankings else "_values"
        matrix_type = "Rankings" if use_rankings else "Values"
        distance_matrices = self.distance_matrices_rankings if use_rankings else self.distance_matrices

        print(f"\nGenerating dendrograms ({matrix_type})...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (metric_name, dist_matrix) in enumerate(distance_matrices.items()):
            ax = axes[idx]
            if np.any(np.isnan(dist_matrix)):
                ax.text(0.5, 0.5, f"NaN in {metric_name} distance", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{metric_name.title()} (Dendrogram, {matrix_type})", fontsize=fontsize + 2)
                continue

            dist_matrix_sym = (dist_matrix + dist_matrix.T) / 2
            np.fill_diagonal(dist_matrix_sym, 0)
            condensed_dist = squareform(dist_matrix_sym)
            linkage_matrix = linkage(condensed_dist, method="average")

            dendrogram(linkage_matrix, labels=self.scenario_names, ax=ax)
            ax.set_title(f"{metric_name.title()} (Dendrogram, {matrix_type})", fontsize=fontsize + 2, fontweight="bold")
            ax.set_xlabel("Scenarios", fontsize=fontsize)
            ax.set_ylabel("Distance", fontsize=fontsize)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize - 1)

        if len(distance_matrices) < len(axes):
            for k in range(len(distance_matrices), len(axes)):
                fig.delaxes(axes[k])

        plt.tight_layout()
        output_path = os.path.join(savepath, f"gsa_dendrograms{suffix}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Dendrograms saved to: {output_path}")

    def plot_difference_heatmaps(self, savepath, fontsize=8, use_rankings=False):
        """Create heatmaps showing parameter-output differences between scenario pairs.
        Non-relevant pairs (where either scenario has value < threshold) shown in grey."""
        matrix_type = "Rankings" if use_rankings else "Values"
        print(f"\nGenerating pairwise difference heatmaps ({matrix_type})...")

        n_scenarios = len(self.scenario_names)
        matrices = self.gsa_ranking_matrices if use_rankings else self.gsa_matrices
        suffix = "_rankings" if use_rankings else "_values"

        # Get parameter and output labels from dictionaries
        param_labels = [self.xlabels_dict_all.get(p, {}).get("latex", p) for p in self.all_params]
        output_labels = [self.ylabels_dict_all.get(o, {}).get("latex", o) for o in self.all_outputs]

        for i in range(n_scenarios):
            for j in range(i + 1, n_scenarios):
                name_i, name_j = self.scenario_names[i], self.scenario_names[j]
                
                # Get value matrices to check relevance
                values_i = self.gsa_matrices[name_i]
                values_j = self.gsa_matrices[name_j]
                
                # Create relevance mask: both must be >= threshold
                relevant_mask = (
                    (~np.isnan(values_i)) & 
                    (~np.isnan(values_j)) & 
                    (values_i >= self.relevance_threshold) & 
                    (values_j >= self.relevance_threshold)
                )
                
                # Calculate difference for comparison matrices
                matrix_i = matrices[name_i]
                matrix_j = matrices[name_j]
                diff_matrix = np.abs(matrix_i - matrix_j)
                
                # Mask out non-relevant pairs
                diff_matrix_masked = np.where(relevant_mask, diff_matrix, np.nan)

                fig_width = max(16, len(self.all_outputs) * 1.5)
                fig_height = max(12, len(self.all_params) * 0.8)
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                
                # Plot heatmap
                hm = sns.heatmap(
                    diff_matrix_masked,
                    xticklabels=output_labels,
                    yticklabels=param_labels,
                    cmap="Reds",
                    ax=ax,
                    cbar_kws={"label": "Absolute Difference"},
                    linewidths=1,
                    linecolor="black",
                    square=True,
                )
                
                # Color non-relevant pairs grey
                for pi in range(diff_matrix.shape[0]):
                    for oi in range(diff_matrix.shape[1]):
                        if not relevant_mask[pi, oi]:
                            ax.add_patch(plt.Rectangle((oi, pi), 1, 1, fill=True, 
                                                     facecolor='lightgray', edgecolor='black', linewidth=1))
                
                ax.set_title(
                    f"GSA Difference: {name_i} vs {name_j} ({matrix_type})\n"
                    f"Grey = non-relevant (< {self.relevance_threshold}) in either scenario",
                    fontsize=fontsize + 4,
                    fontweight="bold",
                    pad=20,
                )
                ax.set_xlabel("Output Features", fontsize=fontsize + 2)
                ax.set_ylabel("Parameters", fontsize=fontsize + 2)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize)
                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize)
                plt.tight_layout()
                output_path = os.path.join(savepath, f"gsa_difference_{i}_{j}{suffix}.png")
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Difference heatmap saved: {name_i} vs {name_j} ({matrix_type})")

    def run_full_comparison(self, savepath):
        os.makedirs(savepath, exist_ok=True)
        self.load_gsa_data()

        # Values analysis
        print("\n=== Running analysis with variance values ===")
        self.compute_distances(use_rankings=False)
        self.generate_report(savepath, use_rankings=False)
        self.plot_distance_heatmaps(savepath, use_rankings=False)
        self.plot_mds_visualization(savepath, use_rankings=False)
        self.plot_dendrogram(savepath, use_rankings=False)
        self.plot_difference_heatmaps(savepath, use_rankings=False)

        # Rankings analysis
        print("\n=== Running analysis with rankings ===")
        self.compute_distances(use_rankings=True)
        self.generate_report(savepath, use_rankings=True)
        self.plot_distance_heatmaps(savepath, use_rankings=True)
        self.plot_mds_visualization(savepath, use_rankings=True)
        self.plot_dendrogram(savepath, use_rankings=True)
        self.plot_difference_heatmaps(savepath, use_rankings=True)

        print("\n" + "=" * 80)
        print("GSA comparison complete!")
        print(f"All outputs saved to: {savepath}")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare GSA results across scenarios")
    parser.add_argument("--scenarios", nargs="+", required=True, help="Paths to scenario folders")
    parser.add_argument("--scenario_names", nargs="+", required=True, help="Names for each scenario")
    parser.add_argument("--xlabels", required=True, help="Path to xlabels file")
    parser.add_argument("--ylabels", required=True, help="Path to ylabels file")
    parser.add_argument("--xlabels_dict", required=True, help="Path to xlabels dictionary")
    parser.add_argument("--ylabels_dict", required=True, help="Path to ylabels dictionary")
    parser.add_argument("--savepath", required=True, help="Output directory for comparison results")
    parser.add_argument("--gsa_mode", default="Si_total", help="GSA mode (default: Si_total)")
    parser.add_argument("--mode", default="max", help="Mode for ranking (default: max)")
    parser.add_argument("--fontsize", type=int, default=10, help="Font size for plots")
    parser.add_argument("--relevance_threshold", type=float, default=0.05, 
                       help="Threshold for parameter relevance (default: 0.05)")

    args = parser.parse_args()

    if len(args.scenarios) != len(args.scenario_names):
        raise ValueError("Number of scenarios must match number of scenario names")

    comparator = GSAComparator(
        scenarios=args.scenarios,
        scenario_names=args.scenario_names,
        xlabels_file=args.xlabels,
        ylabels_file=args.ylabels,
        xlabels_dict=args.xlabels_dict,
        ylabels_dict=args.ylabels_dict,
        gsa_mode=args.gsa_mode,
        mode=args.mode,
        relevance_threshold=args.relevance_threshold,
    )

    comparator.run_full_comparison(args.savepath)


if __name__ == "__main__":
    main()