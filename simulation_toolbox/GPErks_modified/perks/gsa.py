from itertools import combinations
from typing import Callable, Optional, Dict

import alphashape
import copy
import math
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import saltelli, sobol_sequence
from SALib.util import (scale_samples, compute_groups_matrix, _check_groups)
from scipy.special import binom
import scipy.spatial
from sklearn.preprocessing import MinMaxScaler
import tqdm
import warnings


from GPErks_modified.constants import (
    DEFAULT_GSA_CONF_LEVEL,
    DEFAULT_GSA_N,
    DEFAULT_GSA_N_BOOTSTRAP,
    DEFAULT_GSA_N_DRAWS,
    DEFAULT_GSA_SKIP_VALUES,
    DEFAULT_GSA_THRESHOLD,
    DEFAULT_GSA_Z,
)
from GPErks_modified.gp.data.dataset import Dataset
from GPErks_modified.plot.gsa import boxplot, donut, fancy_donut, heatmap, network
from GPErks_modified.plot.options import PlotOptions
from GPErks_modified.plot.plottable import Plottable
from GPErks_modified.train.emulator import GPEmulator
from GPErks_modified.utils.array import get_minmax
from GPErks_modified.perks import history_matching as hm

class SobolGSA_AlphaShape(Plottable):
    def __init__(self, dataset, n, seed=None, alpha_shape_file=None, alpha=0.1):
        super(SobolGSA_AlphaShape, self).__init__()
        self.n = n
        self.seed = seed
        self.alpha_shape_file = alpha_shape_file
        self.alpha = alpha

        self.d = dataset.input_size
        self.index_i = dataset.x_labels
        self.index_ij = [list(c) for c in combinations(self.index_i, 2)]
        self.ylabel = dataset.y_label

        self.minmax = (
            get_minmax(dataset.X_train)
            if (dataset.l_bounds is None and dataset.u_bounds is None)
            else np.hstack(
                (
                    np.array(dataset.l_bounds).reshape(-1, 1),
                    np.array(dataset.u_bounds).reshape(-1, 1),
                )
            )
        )
        print(f"Shape of self.minmax: {self.minmax.shape}")

        self.scaler = MinMaxScaler()
        self.minmax_scaled = self.scaler.fit_transform(self.minmax)  # Normalize bounds

        self.ST = np.zeros((0, self.d), dtype=float)
        self.S1 = np.zeros((0, self.d), dtype=float)
        self.S2 = np.zeros((0, int(binom(self.d, 2))), dtype=float)

        self.ST_std = np.zeros((0, self.d), dtype=float)
        self.S1_std = np.zeros((0, self.d), dtype=float)
        self.S2_std = np.zeros((0, int(binom(self.d, 2))), dtype=float)

    def load_alpha_shape_points(self):
        """Loads and normalizes alpha shape reference points from the file."""
        if self.alpha_shape_file is None:
            return None  # No alpha shape filtering
        points = self.alpha_shape_file
        print("Transforming points...")
        print(f"Shape of points: {points.shape}")

        # Manually normalize using the bounds (self.minmax)
        lower_bounds = self.minmax[:, 0]
        upper_bounds = self.minmax[:, 1]
        
        # Rescale points to [0, 1] using the formula:
        points_scaled = (points - lower_bounds) / (upper_bounds - lower_bounds)

        # Remove duplicate points
        points_scaled = np.unique(points_scaled, axis=0)
        print(f"After removing duplicates, points shape: {points_scaled.shape}")

        # Add small random noise to avoid precision issues
        noise = np.random.normal(0, 1e-8, points_scaled.shape)  # small noise
        points_scaled_noisy = points_scaled + noise

        # Compute alpha shape
        alpha_shape = alphashape.alphashape(points_scaled_noisy, self.alpha)
        
        return alpha_shape

    def is_inside_alpha_shape(self, points, alpha_shape):
        """Checks if points are inside the alpha shape."""
        return np.array([alpha_shape.contains(point) for point in points])

    def assemble_Saltelli_space(self):
        """Generates a Saltelli sample space, filtering samples if an alpha shape is provided."""
        problem = {
            "num_vars": self.d,
            "names": self.index_i,
            "bounds": self.minmax,
        }

        print("Loading alpha shape...")
        alpha_shape = self.load_alpha_shape_points()

        # Generate Saltelli sequence in the original scale
        print("Sampling...")
        all_samples = saltelli.sample(problem, self.n * 8, calc_second_order=True, skip_values=DEFAULT_GSA_SKIP_VALUES)

        print(f"all_samples shape before scaling: {all_samples.shape}")

        # Normalize samples for alpha shape filtering
        all_samples_scaled = self.scaler.transform(all_samples)

        if alpha_shape is not None:
            print("Checking points inside alpha shape...")
            inside_mask = self.is_inside_alpha_shape(all_samples_scaled[:, :self.d], alpha_shape)
            filtered_samples_scaled = all_samples_scaled[inside_mask]

            if len(filtered_samples_scaled) < self.n:
                raise ValueError(f"Not enough samples inside the alpha shape (found {len(filtered_samples_scaled)}, needed {self.n}). "
                                 "Try increasing the base sequence size or relaxing constraints.")

            filtered_samples_scaled = filtered_samples_scaled[:self.n]  # Select first N valid samples
            filtered_samples = self.scaler.inverse_transform(filtered_samples_scaled)  # Rescale back to original range
        else:
            filtered_samples = all_samples[:self.n]  # Use standard GSA if no alpha shape

        return problem, filtered_samples

    def estimate_Sobol_indices_with_emulator(self, emulator, n_draws=10):
        """Estimates Sobol indices using the appropriate sample space."""
        problem, samples = self.assemble_Saltelli_space()
        results = emulator.predict(samples, n_draws=n_draws)
        return results

    def estimate_Sobol_indices_with_simulator(self, f: Callable[[np.ndarray], np.ndarray]):
        """Runs a simulator function to estimate Sobol indices."""
        problem, X = self.assemble_Saltelli_space()
        Y = np.squeeze(f(X)).reshape(1, -1)
        self._estimate_Sobol_indices_from_evaluations(problem, Y)

    def _estimate_Sobol_indices_from_evaluations(self, problem, Y):
        """Computes Sobol indices from function evaluations."""
        for y in tqdm.tqdm(Y):
            S = sobol.analyze(
                problem,
                y,
                calc_second_order=True,
                num_resamples=DEFAULT_GSA_N_BOOTSTRAP,
                conf_level=DEFAULT_GSA_CONF_LEVEL,
                parallel=False,
                n_processors=None,
                seed=self.seed,
            )
            T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S)

            self.ST = np.vstack((self.ST, T_Si["ST"].reshape(1, -1)))
            self.S1 = np.vstack((self.S1, first_Si["S1"].reshape(1, -1)))
            self.S2 = np.vstack((self.S2, np.array(second_Si["S2"]).reshape(1, -1)))

            self.ST_std = np.vstack((self.ST_std, T_Si["ST_conf"].reshape(1, -1) / DEFAULT_GSA_Z))
            self.S1_std = np.vstack((self.S1_std, first_Si["S1_conf"].reshape(1, -1) / DEFAULT_GSA_Z))
            self.S2_std = np.vstack((self.S2_std, (np.array(second_Si["S2_conf"]).reshape(1, -1) / DEFAULT_GSA_Z)))

    def correct_Sobol_indices(self, threshold: float = DEFAULT_GSA_THRESHOLD):
        """Applies a correction to Sobol indices by setting small values to zero."""
        for S in [self.ST, self.S1, self.S2]:
            Q1 = np.percentile(S, q=25, axis=0)
            l = np.where(Q1 < threshold)[0]
            S[:, l] = np.zeros((S.shape[0], len(l)), dtype=float)

    def summary(self):
        """Prints a summary of the computed Sobol indices."""
        self.correct_Sobol_indices()
        df_STi = pd.DataFrame(np.round(np.median(self.ST, axis=0), 6).reshape(-1, 1), index=self.index_i, columns=["STi"])
        df_Si = pd.DataFrame(np.round(np.median(self.S1, axis=0), 6).reshape(-1, 1), index=self.index_i, columns=["Si"])
        df_Sij = pd.DataFrame(np.round(np.median(self.S2, axis=0), 6).reshape(-1, 1), index=["(" + elem[0] + ", " + elem[1] + ")" for elem in self.index_ij], columns=["Sij"])
        print(df_STi)
        print(df_Si)
        print(df_Sij)

    def plot(self, plot_options: PlotOptions = PlotOptions()):
        """Plots results using various visualization methods."""
        self.plot_boxplot()

    def plot_boxplot(self):
        boxplot(self.ST, self.S1, self.S2, self.index_i, self.index_ij, self.ylabel)

    def plot_donut(self):
        donut(self.ST, self.S1, self.index_i, self.ylabel)

    def plot_fancy_donut(self):
        fancy_donut(self.ST, self.S1, self.S2, self.index_i, self.ylabel)

    def plot_heatmap(self):
        heatmap(self.ST, self.S1, self.index_i, self.ylabel)

    def plot_network(self):
        network(self.ST, self.S1, self.S2, self.index_i, self.index_ij, self.ylabel)

class SobolGSA_ConvexHull(Plottable):
    def __init__(self, dataset, n, seed=None, convex_hull_file=None):
        super(SobolGSA_ConvexHull, self).__init__()
        self.n = n
        self.seed = seed
        self.convex_hull_file = convex_hull_file

        self.d = dataset.input_size
        self.index_i = dataset.x_labels
        self.index_ij = [list(c) for c in combinations(self.index_i, 2)]
        self.ylabel = dataset.y_label

        self.minmax = (
            get_minmax(dataset.X_train)
            if (dataset.l_bounds is None and dataset.u_bounds is None)
            else np.hstack(
                (
                    np.array(dataset.l_bounds).reshape(-1, 1),
                    np.array(dataset.u_bounds).reshape(-1, 1),
                )
            )
        )
        print(f"Shape of self.minmax: {self.minmax.shape}")

        self.scaler = MinMaxScaler()
        self.minmax_scaled = self.scaler.fit_transform(self.minmax)  # Normalize bounds

        self.ST = np.zeros((0, self.d), dtype=float)
        self.S1 = np.zeros((0, self.d), dtype=float)
        self.S2 = np.zeros((0, int(binom(self.d, 2))), dtype=float)

        self.ST_std = np.zeros((0, self.d), dtype=float)
        self.S1_std = np.zeros((0, self.d), dtype=float)
        self.S2_std = np.zeros((0, int(binom(self.d, 2))), dtype=float)

    def load_convex_hull_points(self):
        """Loads and normalizes convex hull reference points from the file."""
        if self.convex_hull_file is None:
            return None  # No convex hull filtering
        points =self.convex_hull_file
        print("Transforming points...")
        print(f"Shape of points: {points.shape}")

        # Manually normalize using the bounds (self.minmax)
        lower_bounds = self.minmax[:, 0]
        upper_bounds = self.minmax[:, 1]
        
        # Rescale points to [0, 1] using the formula:
        points_scaled = (points - lower_bounds) / (upper_bounds - lower_bounds)

        # Remove duplicate points
        points_scaled = np.unique(points_scaled, axis=0)
        print(f"After removing duplicates, points shape: {points_scaled.shape}")

            # Add small random noise to avoid precision issues
        noise = np.random.normal(0, 1e-8, points_scaled.shape)  # small noise
        points_scaled_noisy = points_scaled + noise

        # Compute Delaunay triangulation to detect coplanar points

        delaunay = scipy.spatial.Delaunay(points_scaled)
        
        return delaunay

    def is_inside_convex_hull(self, points, hull):
        """Checks if points are inside the convex hull using Delaunay triangulation."""
        return hull.find_simplex(points) >= 0  # Returns True if inside, False otherwise

    def assemble_Saltelli_space(self):
        """Generates a Saltelli sample space, filtering samples if a convex hull is provided."""
        problem = {
            "num_vars": self.d,
            "names": self.index_i,
            "bounds": self.minmax,
        }

        print("Loading convex hull...")
        hull = self.load_convex_hull_points()

        # Generate Saltelli sequence in the original scale
        print("Sampling...")
        all_samples = saltelli.sample(problem, self.n * 8, calc_second_order=True, skip_values=DEFAULT_GSA_SKIP_VALUES)

        print(f"all_samples shape before scaling: {all_samples.shape}")


        # Normalize samples for convex hull filtering
        all_samples_scaled = self.scaler.transform(all_samples)

        if hull is not None:
            print("Checking points inside hull...")
            inside_mask = self.is_inside_convex_hull(all_samples_scaled[:, :self.d], hull)
            filtered_samples_scaled = all_samples_scaled[inside_mask]

            if len(filtered_samples_scaled) < self.n:
                raise ValueError(f"Not enough samples inside the convex hull (found {len(filtered_samples_scaled)}, needed {self.n}). "
                                 "Try increasing the base sequence size or relaxing constraints.")

            filtered_samples_scaled = filtered_samples_scaled[:self.n]  # Select first N valid samples
            filtered_samples = self.scaler.inverse_transform(filtered_samples_scaled)  # Rescale back to original range
        else:
            filtered_samples = all_samples[:self.n]  # Use standard GSA if no convex hull

        return problem, filtered_samples

    def estimate_Sobol_indices_with_emulator(self, emulator, n_draws=10):
        """Estimates Sobol indices using the appropriate sample space."""
        problem, samples = self.assemble_Saltelli_space()
        results = emulator.predict(samples, n_draws=n_draws)
        return results

    def estimate_Sobol_indices_with_simulator(self, f: Callable[[np.ndarray], np.ndarray]):
        """Runs a simulator function to estimate Sobol indices."""
        problem, X = self.assemble_Saltelli_space()
        Y = np.squeeze(f(X)).reshape(1, -1)
        self._estimate_Sobol_indices_from_evaluations(problem, Y)

    def _estimate_Sobol_indices_from_evaluations(self, problem, Y):
        """Computes Sobol indices from function evaluations."""
        for y in tqdm.tqdm(Y):
            S = sobol.analyze(
                problem,
                y,
                calc_second_order=True,
                num_resamples=DEFAULT_GSA_N_BOOTSTRAP,
                conf_level=DEFAULT_GSA_CONF_LEVEL,
                parallel=False,
                n_processors=None,
                seed=self.seed,
            )
            T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S)

            self.ST = np.vstack((self.ST, T_Si["ST"].reshape(1, -1)))
            self.S1 = np.vstack((self.S1, first_Si["S1"].reshape(1, -1)))
            self.S2 = np.vstack((self.S2, np.array(second_Si["S2"]).reshape(1, -1)))

            self.ST_std = np.vstack((self.ST_std, T_Si["ST_conf"].reshape(1, -1) / DEFAULT_GSA_Z))
            self.S1_std = np.vstack((self.S1_std, first_Si["S1_conf"].reshape(1, -1) / DEFAULT_GSA_Z))
            self.S2_std = np.vstack((self.S2_std, (np.array(second_Si["S2_conf"]).reshape(1, -1) / DEFAULT_GSA_Z)))

    def correct_Sobol_indices(self, threshold: float = DEFAULT_GSA_THRESHOLD):
        """Applies a correction to Sobol indices by setting small values to zero."""
        for S in [self.ST, self.S1, self.S2]:
            Q1 = np.percentile(S, q=25, axis=0)
            l = np.where(Q1 < threshold)[0]
            S[:, l] = np.zeros((S.shape[0], len(l)), dtype=float)

    def summary(self):
        """Prints a summary of the computed Sobol indices."""
        self.correct_Sobol_indices()
        df_STi = pd.DataFrame(np.round(np.median(self.ST, axis=0), 6).reshape(-1, 1), index=self.index_i, columns=["STi"])
        df_Si = pd.DataFrame(np.round(np.median(self.S1, axis=0), 6).reshape(-1, 1), index=self.index_i, columns=["Si"])
        df_Sij = pd.DataFrame(np.round(np.median(self.S2, axis=0), 6).reshape(-1, 1), index=["(" + elem[0] + ", " + elem[1] + ")" for elem in self.index_ij], columns=["Sij"])
        print(df_STi)
        print(df_Si)
        print(df_Sij)

    def plot(self, plot_options: PlotOptions = PlotOptions()):
        """Plots results using various visualization methods."""
        self.plot_boxplot()

    def plot_boxplot(self):
        boxplot(self.ST, self.S1, self.S2, self.index_i, self.index_ij, self.ylabel)

    def plot_donut(self):
        donut(self.ST, self.S1, self.index_i, self.ylabel)

    def plot_fancy_donut(self):
        fancy_donut(self.ST, self.S1, self.S2, self.index_i, self.ylabel)

    def plot_heatmap(self):
        heatmap(self.ST, self.S1, self.index_i, self.ylabel)

    def plot_network(self):
        network(self.ST, self.S1, self.S2, self.index_i, self.index_ij, self.ylabel)

class SobolGSA(Plottable):
    def __init__(
        self,
        dataset: Dataset,
        n: int = DEFAULT_GSA_N,
        seed: Optional[int] = None,
    ):
        super(SobolGSA, self).__init__()
        self.n = n
        self.seed = seed

        self.d = dataset.input_size
        self.index_i = dataset.x_labels
        self.index_ij = [list(c) for c in combinations(self.index_i, 2)]
        self.ylabel = dataset.y_label
        self.minmax = (
            get_minmax(dataset.X_train)
            if (dataset.l_bounds is None and dataset.u_bounds is None)
            else np.hstack(
                (
                    np.array(dataset.l_bounds).reshape(-1, 1),
                    np.array(dataset.u_bounds).reshape(-1, 1),
                )
            )
        )

        self.ST = np.zeros((0, self.d), dtype=float)
        self.S1 = np.zeros((0, self.d), dtype=float)
        self.S2 = np.zeros((0, int(binom(self.d, 2))), dtype=float)

        self.ST_std = np.zeros((0, self.d), dtype=float)
        self.S1_std = np.zeros((0, self.d), dtype=float)
        self.S2_std = np.zeros((0, int(binom(self.d, 2))), dtype=float)

    def assemble_Saltelli_space(self):
        problem = {
            "num_vars": self.d,
            "names": self.index_i,
            "bounds": self.minmax,
        }
        X = saltelli.sample(
            problem,
            self.n,
            calc_second_order=True,
            skip_values=DEFAULT_GSA_SKIP_VALUES,
        )
        return problem, X

    def estimate_Sobol_indices_with_emulator(
        self,
        emulator: GPEmulator,
        n_draws: int = DEFAULT_GSA_N_DRAWS,
    ):
        print("Assembling...")
        problem, X = self.assemble_Saltelli_space()
        print("Sampling...")
        Y = emulator.sample(X, n_draws)
        print("Estimating..")
        self._estimate_Sobol_indices_from_evaluations(problem, Y)

    def estimate_Sobol_indices_with_simulator(
        self,
        f: Callable[[np.ndarray], np.ndarray],
    ):
        problem, X = self.assemble_Saltelli_space()
        Y = np.squeeze(f(X)).reshape(1, -1)
        self._estimate_Sobol_indices_from_evaluations(problem, Y)

    def _estimate_Sobol_indices_from_evaluations(self, problem, Y):
        for y in tqdm.tqdm(Y):
            S = sobol.analyze(
                problem,
                y,
                calc_second_order=True,
                num_resamples=DEFAULT_GSA_N_BOOTSTRAP,
                conf_level=DEFAULT_GSA_CONF_LEVEL,
                parallel=False,
                n_processors=None,
                seed=self.seed,
            )
            T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S)

            self.ST = np.vstack((self.ST, T_Si["ST"].reshape(1, -1)))
            self.S1 = np.vstack((self.S1, first_Si["S1"].reshape(1, -1)))
            self.S2 = np.vstack(
                (self.S2, np.array(second_Si["S2"]).reshape(1, -1))
            )

            self.ST_std = np.vstack(
                (self.ST_std, T_Si["ST_conf"].reshape(1, -1) / DEFAULT_GSA_Z)
            )
            self.S1_std = np.vstack(
                (
                    self.S1_std,
                    first_Si["S1_conf"].reshape(1, -1) / DEFAULT_GSA_Z,
                )
            )
            self.S2_std = np.vstack(
                (
                    self.S2_std,
                    (
                        np.array(second_Si["S2_conf"]).reshape(1, -1)
                        / DEFAULT_GSA_Z
                    ),
                )
            )

    def correct_Sobol_indices(self, threshold: float = DEFAULT_GSA_THRESHOLD):
        for S in [self.ST, self.S1, self.S2]:
            Q1 = np.percentile(S, q=25, axis=0)
            l = np.where(Q1 < threshold)[0]
            S[:, l] = np.zeros((S.shape[0], len(l)), dtype=float)

    def summary(self):
        self.correct_Sobol_indices()
        df_STi = pd.DataFrame(
            data=np.round(np.median(self.ST, axis=0), 6).reshape(-1, 1),
            index=self.index_i,
            columns=["STi"],
        )
        df_Si = pd.DataFrame(
            data=np.round(np.median(self.S1, axis=0), 6).reshape(-1, 1),
            index=self.index_i,
            columns=["Si"],
        )
        df_Sij = pd.DataFrame(
            data=np.round(np.median(self.S2, axis=0), 6).reshape(-1, 1),
            index=[
                "(" + elem[0] + ", " + elem[1] + ")" for elem in self.index_ij
            ],
            columns=["Sij"],
        )
        print(df_STi)
        print(df_Si)
        print(df_Sij)

    def plot(self, plot_options: PlotOptions = PlotOptions()):
        self.plot_boxplot()

    def plot_boxplot(self):
        boxplot(
            self.ST, self.S1, self.S2, self.index_i, self.index_ij, self.ylabel
        )

    def plot_donut(self):
        donut(self.ST, self.S1, self.index_i, self.ylabel)

    def plot_fancy_donut(self):
        fancy_donut(self.ST, self.S1, self.S2, self.index_i, self.ylabel)

    def plot_heatmap(self):
        heatmap(self.ST, self.S1, self.index_i, self.ylabel)

    def plot_network(self):
        network(
            self.ST, self.S1, self.S2, self.index_i, self.index_ij, self.ylabel
        )


class SobolGSA_NIMP(Plottable):
    def __init__(
        self,
        dataset: Dataset,
        n: int = DEFAULT_GSA_N,
        seed: Optional[int] = None,
    ):
        super(SobolGSA_NIMP, self).__init__()
        self.n = n
        self.seed = seed

        self.d = dataset.input_size
        self.index_i = dataset.x_labels
        self.index_ij = [list(c) for c in combinations(self.index_i, 2)]
        self.ylabel = dataset.y_label
        self.minmax = (
            get_minmax(dataset.X_train)
            if (dataset.l_bounds is None and dataset.u_bounds is None)
            else np.hstack(
                (
                    np.array(dataset.l_bounds).reshape(-1, 1),
                    np.array(dataset.u_bounds).reshape(-1, 1),
                )
            )
        )

        self.ST = np.zeros((0, self.d), dtype=float)
        self.S1 = np.zeros((0, self.d), dtype=float)
        self.S2 = np.zeros((0, int(binom(self.d, 2))), dtype=float)

        self.ST_std = np.zeros((0, self.d), dtype=float)
        self.S1_std = np.zeros((0, self.d), dtype=float)
        self.S2_std = np.zeros((0, int(binom(self.d, 2))), dtype=float)
    

    def sample_NIMP(self, problem: Dict, N: int, calc_second_order: bool = True,
            skip_values: int = 0, wave_file : str = None, wave_features_idx=None):
        """Generates model inputs using Saltelli's extension of the Sobol' sequence.

        Returns a NumPy matrix containing the model inputs using Saltelli's sampling
        scheme. Saltelli's scheme extends the Sobol' sequence in a way to reduce
        the error rates in the resulting sensitivity index calculations. If
        `calc_second_order` is False, the resulting matrix has ``N * (D + 2)``
        rows, where ``D`` is the number of parameters. If `calc_second_order` is True,
        the resulting matrix has ``N * (2D + 2)`` rows. These model inputs are
        intended to be used with :func:`SALib.analyze.sobol.analyze`.

        If `skip_values` is > 0, raises a UserWarning in cases where sample sizes may 
        be sub-optimal. The convergence properties of the Sobol' sequence requires
        ``N < skip_values`` and that both `N` and `skip_values` are base 2 
        (e.g., ``N = 2^n``). See discussion in [4] for context and information.

        If skipping values, one recommendation is that the largest possible `n` such that
        ``(2^n)-1 <= N`` is skipped (see [5]).

        Parameters
        ----------
        problem : dict
            The problem definition
        N : int
            The number of samples to generate.
            Must be an exponent of 2 and < `skip_values`.
        calc_second_order : bool
            Calculate second-order sensitivities (default True)
        skip_values : int
            Number of points in Sobol' sequence to skip, ideally a value of base 2
            (default 0, see Owen [3] and Discussion [4])


        References
        ----------
        .. [1] Saltelli, A., 2002.
            Making best use of model evaluations to compute sensitivity indices.
            Computer Physics Communications 145, 280–297.
            https://doi.org/10.1016/S0010-4655(02)00280-1

        .. [2] Sobol', I.M., 2001.
            Global sensitivity indices for nonlinear mathematical models and
            their Monte Carlo estimates.
            Mathematics and Computers in Simulation,
            The Second IMACS Seminar on Monte Carlo Methods 55, 271–280.
            https://doi.org/10.1016/S0378-4754(00)00270-6

        .. [3] Owen, A. B., 2020.
            On dropping the first Sobol' point.
            arXiv:2008.08051 [cs, math, stat].
            Available at: http://arxiv.org/abs/2008.08051 (Accessed: 20 April 2021).

        .. [4] Discussion: https://github.com/scipy/scipy/pull/10844
            https://github.com/scipy/scipy/pull/10844#issuecomment-673029539
        
        .. [5] Johnson, S. G. 
            Sobol.jl: The Sobol module for Julia
            https://github.com/stevengj/Sobol.jl
            
        """
        # bit-shift test to check if `N` == 2**n
        if not ((N & (N-1) == 0) and (N != 0 and N-1 != 0)):
            msg = f"""
            Convergence properties of the Sobol' sequence is only valid if
            `N` ({N}) is equal to `2^n`.
            """
            warnings.warn(msg)

        if skip_values > 0:
            M = skip_values
            if not ((M & (M-1) == 0) and (M != 0 and M-1 != 0)):
                msg = f"""
                Convergence properties of the Sobol' sequence is only valid if
                `skip_values` ({M}) is equal to `2^m`.
                """
                warnings.warn(msg)

            n_exp = int(math.log(N, 2))
            m_exp = int(math.log(M, 2))
            if n_exp >= m_exp:
                msg = f"Convergence may not be valid as 2^{n_exp} ({N}) is >= 2^{m_exp} ({M})."
                warnings.warn(msg)

        D = problem['num_vars']
        groups = _check_groups(problem)

        if not groups:
            Dg = problem['num_vars']
        else:
            G, group_names = compute_groups_matrix(groups)
            Dg = len(set(group_names))


        ###### The base sequence here will be only based on the NIMP region
        # Create base sequence - could be any type of sampling
        n_viable = 0
        initial_N = N

        base_sequence_initial = sobol_sequence.sample(N + skip_values, 2 * D)

        while n_viable < initial_N:
            A = base_sequence_initial[:,:D]
            B = base_sequence_initial[:,D:]
            A_rescaled = copy.deepcopy(A)
            B_rescaled = copy.deepcopy(B)

            for i in range(D):
                A_rescaled[:,i] = A_rescaled[:,i]*(problem['bounds'][i,1]-problem['bounds'][i,0])+problem['bounds'][i,0]
                B_rescaled[:,i] = B_rescaled[:,i]*(problem['bounds'][i,1]-problem['bounds'][i,0])+problem['bounds'][i,0]    

            W = hm.Wave()
            W.load(wave_file) 

            print('-----------------------------')
            print('Loading emulators for wave...')
            print('-----------------------------')      

            emulator_w = []
            for idx_w in wave_features_idx:
                loadpath_wave = wave_gpepath + str(idx_w) + "/"
                X_train_w = np.loadtxt(loadpath_wave + "X_train.txt", dtype=np.float64)
                y_train_w = np.loadtxt(loadpath_wave + "y_train.txt", dtype=np.float64)
                emul_w = GPEmul.load(X_train_w, y_train_w, loadpath=loadpath_wave)
                emulator_w.append(emul_w)
            
            W.emulator = emulator_w   

        #################################################
        if calc_second_order:
            saltelli_sequence = np.zeros([(2 * Dg + 2) * N, D])
        else:
            saltelli_sequence = np.zeros([(Dg + 2) * N, D])
        index = 0

        for i in range(skip_values, N + skip_values):

            # Copy matrix "A"
            for j in range(D):
                saltelli_sequence[index, j] = base_sequence[i, j]

            index += 1

            # Cross-sample elements of "B" into "A"
            for k in range(Dg):
                for j in range(D):
                    if (not groups and j == k) or (groups and group_names[k] == groups[j]):
                        saltelli_sequence[index, j] = base_sequence[i, j + D]
                    else:
                        saltelli_sequence[index, j] = base_sequence[i, j]

                index += 1

            # Cross-sample elements of "A" into "B"
            # Only needed if you're doing second-order indices (true by default)
            if calc_second_order:
                for k in range(Dg):
                    for j in range(D):
                        if (not groups and j == k) or (groups and group_names[k] == groups[j]):
                            saltelli_sequence[index, j] = base_sequence[i, j]
                        else:
                            saltelli_sequence[index, j] = base_sequence[i, j + D]

                    index += 1

            # Copy matrix "B"
            for j in range(D):
                saltelli_sequence[index, j] = base_sequence[i, j + D]

            index += 1

        saltelli_sequence = scale_samples(saltelli_sequence, problem)
        return saltelli_sequence


    def assemble_Saltelli_space(self, wave_file):
        problem = {
            "num_vars": self.d,
            "names": self.index_i,
            "bounds": self.minmax,
        }

        print(f"Wave file is {wave_file}")

        ##### We change here to use NIMP
        X = self.sample_NIMP(
            problem = problem,
            N = self.n,
            calc_second_order=True,
            skip_values=DEFAULT_GSA_SKIP_VALUES,
            wave_file = wave_file
        )
        return problem, X

    def estimate_Sobol_indices_with_emulator(
        self,
        emulator: GPEmulator,
        n_draws: int = DEFAULT_GSA_N_DRAWS,
        wave_file: str = None
    ):
        print("Assembling...")
        problem, X = self.assemble_Saltelli_space(wave_file=wave_file)
        print("Sampling...")
        Y = emulator.sample(X, n_draws)
        print("Estimating..")
        self._estimate_Sobol_indices_from_evaluations(problem, Y)

    def estimate_Sobol_indices_with_simulator(
        self,
        f: Callable[[np.ndarray], np.ndarray],
    ):
        problem, X = self.assemble_Saltelli_space()
        Y = np.squeeze(f(X)).reshape(1, -1)
        self._estimate_Sobol_indices_from_evaluations(problem, Y)

    def _estimate_Sobol_indices_from_evaluations(self, problem, Y):
        for y in tqdm.tqdm(Y):
            S = sobol.analyze(
                problem,
                y,
                calc_second_order=True,
                num_resamples=DEFAULT_GSA_N_BOOTSTRAP,
                conf_level=DEFAULT_GSA_CONF_LEVEL,
                parallel=False,
                n_processors=None,
                seed=self.seed,
            )
            T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S)

            self.ST = np.vstack((self.ST, T_Si["ST"].reshape(1, -1)))
            self.S1 = np.vstack((self.S1, first_Si["S1"].reshape(1, -1)))
            self.S2 = np.vstack(
                (self.S2, np.array(second_Si["S2"]).reshape(1, -1))
            )

            self.ST_std = np.vstack(
                (self.ST_std, T_Si["ST_conf"].reshape(1, -1) / DEFAULT_GSA_Z)
            )
            self.S1_std = np.vstack(
                (
                    self.S1_std,
                    first_Si["S1_conf"].reshape(1, -1) / DEFAULT_GSA_Z,
                )
            )
            self.S2_std = np.vstack(
                (
                    self.S2_std,
                    (
                        np.array(second_Si["S2_conf"]).reshape(1, -1)
                        / DEFAULT_GSA_Z
                    ),
                )
            )

    def correct_Sobol_indices(self, threshold: float = DEFAULT_GSA_THRESHOLD):
        for S in [self.ST, self.S1, self.S2]:
            Q1 = np.percentile(S, q=25, axis=0)
            l = np.where(Q1 < threshold)[0]
            S[:, l] = np.zeros((S.shape[0], len(l)), dtype=float)

    def summary(self):
        self.correct_Sobol_indices()
        df_STi = pd.DataFrame(
            data=np.round(np.median(self.ST, axis=0), 6).reshape(-1, 1),
            index=self.index_i,
            columns=["STi"],
        )
        df_Si = pd.DataFrame(
            data=np.round(np.median(self.S1, axis=0), 6).reshape(-1, 1),
            index=self.index_i,
            columns=["Si"],
        )
        df_Sij = pd.DataFrame(
            data=np.round(np.median(self.S2, axis=0), 6).reshape(-1, 1),
            index=[
                "(" + elem[0] + ", " + elem[1] + ")" for elem in self.index_ij
            ],
            columns=["Sij"],
        )
        print(df_STi)
        print(df_Si)
        print(df_Sij)

    def plot(self, plot_options: PlotOptions = PlotOptions()):
        self.plot_boxplot()

    def plot_boxplot(self):
        boxplot(
            self.ST, self.S1, self.S2, self.index_i, self.index_ij, self.ylabel
        )

    def plot_donut(self):
        donut(self.ST, self.S1, self.index_i, self.ylabel)

    def plot_fancy_donut(self):
        fancy_donut(self.ST, self.S1, self.S2, self.index_i, self.ylabel)

    def plot_heatmap(self):
        heatmap(self.ST, self.S1, self.index_i, self.ylabel)

    def plot_network(self):
        network(
            self.ST, self.S1, self.S2, self.index_i, self.index_ij, self.ylabel
        )
