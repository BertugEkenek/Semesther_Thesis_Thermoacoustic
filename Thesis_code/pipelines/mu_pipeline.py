# pipeline/mu_pipeline.py

import numpy as np

from utils import logger
from dataio.dataloader import load_data, reshape_EV_trajectories, load_txt_solutions
from pipelines.prep_work import prepare_mu
from fitting import linear_fit as linfit
from fitting import second_order_fit as sofit


class MUFITPipeline:
    """
    Pipeline responsible for:
      - loading eigenvalue data (MAT or TXT)
      - reshaping trajectories
      - building μ fits (first- and second-order)
    """

    def __init__(
        self,
        config: object,
        data_path: str,
        txt_solution_path: str,
        order: int,
        use_only_acoustic: bool,
        use_txt_solutions: bool,
        enforce_symmetry: bool,
    ):
        self.data_path = data_path
        self.order = order
        self.config = config
        self.logger = logger
        self.use_only_acoustic = use_only_acoustic
        self.use_txt_solutions = use_txt_solutions
        self.txt_solution_path = txt_solution_path
        self.enforce_symmetry = enforce_symmetry

        self.EV_trajectories_stacked = None
        self.n = None
        self.R = None
        self.EV0 = None
        self.min_size = None
        self.EV_trajectories = None
        self.EV_trajectories_reshaped = None

    # --------------------------------------------------------------
    def load_all_data(self):
        if self.use_txt_solutions:
            self.logger.info(
                f"Loading eigenvalue solutions from {self.txt_solution_path}"
            )
            (
                self.n,
                self.R,
                self.EV0,
                self.EV_trajectories,
                _,
                _,
                self.min_size,
            ) = load_txt_solutions(self.txt_solution_path)

        else:
            self.logger.info(f"Loading main data from {self.data_path}")
            (
                self.n,
                self.R,
                self.EV0,
                self.EV_trajectories,
                _,
                _,
                self.min_size,
            ) = load_data(
                data_path=self.data_path,
                show_position=False,
                show_max_min=False,
            )

        self.EV_trajectories_reshaped = reshape_EV_trajectories(
            self.EV_trajectories, self.min_size
        )
        self.logger.info("Data loading completed.")

    # --------------------------------------------------------------
    def prepare(self):
        # Convert to numeric. For MAT-data, this yields shape [R, τ, n]
        # where each entry is a single complex eigenvalue (acoustic),
        # because prepare_mu picks the first eigenvalue in the object entry.
        self.EV_trajectories_stacked = prepare_mu(
            self.EV_trajectories_reshaped, self.min_size
        )

        if self.use_only_acoustic:
            self.logger.info(
                "Using only acoustic part (first eigenvalue per [R, τ, n])"
            )
            # no extra slicing needed

    # --------------------------------------------------------------
    def find_mu_fit(self, config, tau):
        self.logger.info("Computing first-order μ fit")

        EV0 = np.hstack(self.EV0).ravel()
        n = self.n
        stacked = self.EV_trajectories_stacked  # shape [R, τ, n]

        num_R = stacked.shape[0]
        mu_array = np.zeros((num_R, 2))

        for i in range(num_R):
            w_big = stacked[i, 0, :].imag
            sigma_big = stacked[i, 0, :].real

            s_1 = EV0[i]
            sigma0 = s_1.real
            w0 = s_1.imag

            A = linfit.build_A(n, tau, w_big, sigma_big, sigma0, w0, True)
            b = linfit.build_b(config, n, tau, w_big, sigma_big, s_1, sigma0, w0, True)

            mu = linfit.regression(A, b)
            mu_array[i, 0] = mu[0]
            mu_array[i, 1] = mu[1]

        self.logger.info("First-order μ fit completed")
        return mu_array, self.R, EV0

    # --------------------------------------------------------------
    def find_mu_fit_second_order(
        self,
        config: object,
        tau: float,
        order: int,
        use_only_acoustic: bool,
        enforce_symmetry: bool,
    ):
        """
        Compute second-order *global* μ values (one global parameter vector per R).

        Returns
        -------
        mu_global : np.ndarray
            Shape (num_R, p_dim), where p_dim = 6 if enforce_symmetry else 8.
        R : np.ndarray
            The R vector from the data.
        EV0_flat : np.ndarray
            The base eigenvalues (s_1) for each R (flattened).
        """
        self.logger.info("Computing second-order μ fit (global μ per R)")

        # Flatten EV0 exactly like the original code did
        EV0_flat = np.hstack(self.EV0).ravel()
        n = self.n
        stacked = self.EV_trajectories_stacked  # shape [num_R, τ, n] after prepare()
        num_R = stacked.shape[0]

        mu_global_list = []

        for i in range(num_R):
            # s_1 for THIS R index
            s_1 = EV0_flat[i]

            # s_2 is provided via config.w[1] for now
            s_2 = config.w[1]

            # Acoustic trajectory over n for this R (τ index = 0)
            w_big = stacked[i, 0, :].imag
            sigma_big = stacked[i, 0, :].real

            # Only care about global μ parameters (mu_opt)
            _, mu_opt, info = sofit.mu_array_stacked(
                config,
                n,
                tau,
                w_big,
                sigma_big,
                s_1,
                s_2,
                use_only_acoustic,
                enforce_symmetry,
            )

            mu_global_list.append(mu_opt)
            self.logger.debug(
                f"R[{i}]={self.R[i]}: s1={s_1}, global μ={mu_opt}, cost={info['global_cost']}"
            )

        mu_global = np.vstack(mu_global_list)  # shape (num_R, p_dim)
        self.logger.info("Second-order global μ fit completed")

        return mu_global, self.R, EV0_flat
