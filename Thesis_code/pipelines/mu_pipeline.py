import numpy as np

from utils import logger
from dataio.dataloader import load_data, reshape_EV_trajectories, load_txt_solutions
from pipelines.prep_work import prepare_mu
from fitting import linear_fit as linfit
from fitting import second_order_fit as sofit


class MUFITPipeline:
    """
    Pipeline responsible for:
      - loading eigenvalue data (MAT or TXT) for acoustic branch 1
      - optionally loading a second .mat file for acoustic branch 2
      - reshaping and converting EV trajectories to numeric form
      - building μ-fits (first- and second-order).

    Multi-branch behavior:
      - Branch 1 is the reference acoustic branch.
      - Optionally, branch 2 is included as additional constraints
        when fitting second-order μ, but the dimensionality of μ
        (6 or 8 real parameters) is unchanged.

    SECOND-ORDER (NEW)
    -------------------
    The second-order μ-fit now uses a *global per-R* nonlinear
    least-squares solve:

      - For each R, all n-samples (and optional branches) are
        stacked into one big system M μ - b = 0.
      - One bounded nonlinear LSQ problem is solved per R.
      - There are no per-n local μ-solves anymore.

    The function find_mu_fit_second_order still returns:
      - mu_array : shape (num_R, p_dim) with p_dim=6 or 8
      - R        : 1D array of R values
      - EV0_flat : 1D array of base eigenvalues (branch 1)
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
        # --- multi-branch options ---
        num_acoustic_branches: int = 1,
        fit_branches=None,
        branch2_data_path: str | None = None,
    ):
        # original args
        self.data_path = data_path
        self.txt_solution_path = txt_solution_path
        self.order = order
        self.config = config
        self.logger = logger
        self.use_only_acoustic = use_only_acoustic
        self.use_txt_solutions = use_txt_solutions
        self.enforce_symmetry = enforce_symmetry

        # main (branch-1) data storage
        self.EV_trajectories_stacked = None
        self.n = None
        self.R = None
        self.EV0 = None
        self.min_size = None
        self.EV_trajectories = None
        self.EV_trajectories_reshaped = None

        # branch-2 configuration and storage
        self.num_acoustic_branches = int(num_acoustic_branches)
        self.fit_branches = fit_branches if fit_branches is not None else [1]
        self.branch2_data_path = branch2_data_path

        self.EV0_branch2 = None
        self.EV_trajectories_branch2 = None
        self.EV_trajectories_branch2_reshaped = None
        self.EV_trajectories_branch2_stacked = None
        self.min_size_branch2 = None

    # --------------------------------------------------------------
    def load_all_data(self):
        """
        Load eigenvalue data for branch 1 and (optionally) branch 2.

        Branch 1:
          - Source controlled by use_txt_solutions flag (TXT vs MAT).
        Branch 2:
          - Always loaded from a MAT file via load_data, if requested.

        The method also reshapes each trajectory set to a uniform
        min_size using reshape_EV_trajectories.
        """

        # -----------------
        # Branch 1
        # -----------------
        if self.use_txt_solutions:
            self.logger.info(
                f"Loading eigenvalue solutions from TXT: {self.txt_solution_path}"
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
            self.logger.info(f"Loading branch-1 data from MAT: {self.data_path}")
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

        # -----------------
        # Optional branch 2
        # -----------------
        if self.num_acoustic_branches >= 2:
            if self.branch2_data_path is None:
                raise ValueError(
                    "num_acoustic_branches >= 2, but branch2_data_path is not set."
                )

            self.logger.info(
                f"Loading branch-2 data from MAT: {self.branch2_data_path}"
            )
            (
                n2,
                R2,
                self.EV0_branch2,
                self.EV_trajectories_branch2,
                _,
                _,
                self.min_size_branch2,
            ) = load_data(
                data_path=self.branch2_data_path,
                show_position=False,
                show_max_min=False,
            )

            # Consistency check
            if not np.allclose(self.n, n2):
                raise ValueError(
                    "Branch 1 and branch 2 have different n-grids; "
                    "this is not supported."
                )
            if not np.allclose(self.R, R2):
                raise ValueError(
                    "Branch 1 and branch 2 have different R-grids; "
                    "this is not supported."
                )

            self.EV_trajectories_branch2_reshaped = reshape_EV_trajectories(
                self.EV_trajectories_branch2, self.min_size_branch2
            )

            self.logger.info(
                "Branch-2 data loaded and reshaped "
                f"(min_size_branch2={self.min_size_branch2})."
            )
        else:
            self.logger.info("Single-branch configuration (num_acoustic_branches = 1).")

        self.logger.info("Data loading completed.")

    # --------------------------------------------------------------
    def prepare(self):
        """
        Convert object-typed EV trajectories into numeric complex arrays.

        For MAT-data, this yields shape [R, τ, n] after prepare_mu,
        where each entry is the (complex) acoustic eigenvalue for the
        corresponding (R, τ, n) index.

        For branch 2, if present, the same conversion is applied.
        """

        # Branch 1
        self.EV_trajectories_stacked = prepare_mu(
            self.EV_trajectories_reshaped, self.min_size
        )

        if self.use_only_acoustic:
            self.logger.info(
                "Using only acoustic part (first eigenvalue per [R, τ, n]) "
                "for branch 1 (legacy behavior)."
            )
            # No extra slicing needed; prepare_mu already takes first eigenvalue.

        # Branch 2
        if (
            self.num_acoustic_branches >= 2
            and self.EV_trajectories_branch2_reshaped is not None
        ):
            self.EV_trajectories_branch2_stacked = prepare_mu(
                self.EV_trajectories_branch2_reshaped, self.min_size_branch2
            )
            self.logger.info(
                "Branch-2 trajectories converted to numeric complex array."
            )
        else:
            self.EV_trajectories_branch2_stacked = None

    # --------------------------------------------------------------
    def find_mu_fit(self, config, tau):
        """
        First-order μ-fit (legacy), per R.

        Uses only branch-1 trajectories and calls the linear
        first-order fitter in fitting.linear_fit.
        """

        self.logger.info("Computing first-order μ fit (branch 1 only)")

        EV0 = np.hstack(self.EV0).ravel()
        n = self.n
        stacked = self.EV_trajectories_stacked  # [num_R, τ, n]

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
        Second-order μ-fit (per R), multi-branch aware.

        For each R[i], it:
          - chooses s_1 (branch-1 EV0) and s_2 (branch-2 EV0 if
            available, otherwise from config.w_R_table or config.w[1]),
          - stacks all n-samples (and optional extra branches)
            into a global M μ - b system,
          - calls second_order_fit.solve_mu_per_R to solve a
            bounded nonlinear LSQ for p (μ-parameters).

        Returns
        -------
        mu_array : np.ndarray
            Shape (num_R, p_dim), where p_dim = 6 (symmetric) or 8.
            Each row is the μ-parameter vector p for that R.
        R : np.ndarray
            R-grid used for fitting.
        EV0_flat_1 : np.ndarray
            Flattened base eigenvalues (branch 1).
        """

        if 1 not in self.fit_branches:
            raise ValueError(
                "fit_branches must contain branch 1. "
                "Branch 1 defines the reference mode s_1."
            )

        self.logger.info("Computing second-order μ fit (per R, global, multi-branch)")

        # Flatten EV0 for branch 1
        EV0_flat_1 = np.hstack(self.EV0).ravel()
        n = self.n
        stacked1 = self.EV_trajectories_stacked  # [num_R, τ, n]
        num_R = stacked1.shape[0]

        # Optional branch-2 EV0 flattening
        EV0_flat_2 = None
        if self.EV0_branch2 is not None:
            try:
                EV0_flat_2 = np.hstack(self.EV0_branch2).ravel()
            except Exception as exc:
                self.logger.error(f"Failed to flatten EV0_branch2: {exc}")
                raise

        p_list = []

        for i in range(num_R):
            # ------------------------
            # Base modes for this R[i]
            # ------------------------
            s_1 = EV0_flat_1[i]

            # Prefer EV0_branch2 as s_2 if available
            if EV0_flat_2 is not None:
                s_2 = EV0_flat_2[i]
            else:
                R_val = float(self.R[i])
                try:
                    s_2 = config.w_R_table[R_val]
                except Exception:
                    s_2 = config.w[1]
                    self.logger.debug(
                        f"Using config.w[1] as s_2 fallback at R={R_val}."
                    )

            # Acoustic trajectory for branch 1 at this R (τ index = 0)
            w_big_1 = stacked1[i, 0, :].imag
            sigma_big_1 = stacked1[i, 0, :].real

            w_big_1 = np.asarray(w_big_1, float)
            sigma_big_1 = np.asarray(sigma_big_1, float)

            # -----------------------------------
            # Optional extra branch: branch 2
            # -----------------------------------
            extra_branches = []

            if (
                2 in self.fit_branches
                and self.num_acoustic_branches >= 2
                and self.EV_trajectories_branch2_stacked is not None
            ):
                w_big_2 = self.EV_trajectories_branch2_stacked[i, 0, :].imag
                sigma_big_2 = self.EV_trajectories_branch2_stacked[i, 0, :].real

                w_big_2 = np.asarray(w_big_2, float)
                sigma_big_2 = np.asarray(sigma_big_2, float)

                extra_branches.append(
                    {
                        "w_big": w_big_2,
                        "sigma_big": sigma_big_2,
                        "s_ref": s_2,
                        "s_other": s_1,
                    }
                )

            # -----------------------------------
            # Global per-R μ-fit using second-order model
            # -----------------------------------
            p_opt, info = sofit.solve_mu_per_R(
                config=config,
                n=n,
                tau=tau,
                w_big=w_big_1,
                sigma_big=sigma_big_1,
                s_1=s_1,
                s_2=s_2,
                use_only_acoustic=use_only_acoustic,
                enforce_symmetry=enforce_symmetry,
                weights=None,
                quiet=False,
                extra_branches=extra_branches if extra_branches else None,
            )

            p_list.append(p_opt)

            self.logger.info(
                f"R[{i}]={self.R[i]}: s1={s_1}, s2={s_2}, p={p_opt}, "
                f"cost={info.get('global_cost', np.nan):.3e}, "
                f"cond_raw={info.get('cond_M', np.nan):.2e}, "
                f"cond_scaled={info.get('cond_M_scaled', np.nan):.2e}, "
                f"||res||={info.get('residual_global_norm', np.nan):.3e}"
            )

        mu_array = np.vstack(p_list)  # (num_R, p_dim)
        self.logger.info("Second-order μ fit completed (per-R global solve)")

        return mu_array, self.R, EV0_flat_1
import numpy as np

from utils import logger
from dataio.dataloader import load_data, reshape_EV_trajectories, load_txt_solutions
from pipelines.prep_work import prepare_mu
from fitting import linear_fit as linfit
from fitting import second_order_fit as sofit


class MUFITPipeline:
    """
    Pipeline responsible for:
      - loading eigenvalue data (MAT or TXT) for acoustic branch 1
      - optionally loading a second .mat file for acoustic branch 2
      - reshaping and converting EV trajectories to numeric form
      - building μ-fits (first- and second-order).

    Multi-branch behavior:
      - Branch 1 is the reference acoustic branch.
      - Optionally, branch 2 is included as additional constraints
        when fitting second-order μ, but the dimensionality of μ
        (6 or 8 real parameters) is unchanged.

    SECOND-ORDER (NEW)
    -------------------
    The second-order μ-fit now uses a *global per-R* nonlinear
    least-squares solve:

      - For each R, all n-samples (and optional branches) are
        stacked into one big system M μ - b = 0.
      - One bounded nonlinear LSQ problem is solved per R.
      - There are no per-n local μ-solves anymore.

    The function find_mu_fit_second_order still returns:
      - mu_array : shape (num_R, p_dim) with p_dim=6 or 8
      - R        : 1D array of R values
      - EV0_flat : 1D array of base eigenvalues (branch 1)
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
        # --- multi-branch options ---
        num_acoustic_branches: int = 1,
        fit_branches=None,
        branch2_data_path: str | None = None,
    ):
        # original args
        self.data_path = data_path
        self.txt_solution_path = txt_solution_path
        self.order = order
        self.config = config
        self.logger = logger
        self.use_only_acoustic = use_only_acoustic
        self.use_txt_solutions = use_txt_solutions
        self.enforce_symmetry = enforce_symmetry

        # main (branch-1) data storage
        self.EV_trajectories_stacked = None
        self.n = None
        self.R = None
        self.EV0 = None
        self.min_size = None
        self.EV_trajectories = None
        self.EV_trajectories_reshaped = None

        # branch-2 configuration and storage
        self.num_acoustic_branches = int(num_acoustic_branches)
        self.fit_branches = fit_branches if fit_branches is not None else [1]
        self.branch2_data_path = branch2_data_path

        self.EV0_branch2 = None
        self.EV_trajectories_branch2 = None
        self.EV_trajectories_branch2_reshaped = None
        self.EV_trajectories_branch2_stacked = None
        self.min_size_branch2 = None

    # --------------------------------------------------------------
    def load_all_data(self):
        """
        Load eigenvalue data for branch 1 and (optionally) branch 2.

        Branch 1:
          - Source controlled by use_txt_solutions flag (TXT vs MAT).
        Branch 2:
          - Always loaded from a MAT file via load_data, if requested.

        The method also reshapes each trajectory set to a uniform
        min_size using reshape_EV_trajectories.
        """

        # -----------------
        # Branch 1
        # -----------------
        if self.use_txt_solutions:
            self.logger.info(
                f"Loading eigenvalue solutions from TXT: {self.txt_solution_path}"
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
            self.logger.info(f"Loading branch-1 data from MAT: {self.data_path}")
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

        # -----------------
        # Optional branch 2
        # -----------------
        if self.num_acoustic_branches >= 2:
            if self.branch2_data_path is None:
                raise ValueError(
                    "num_acoustic_branches >= 2, but branch2_data_path is not set."
                )

            self.logger.info(
                f"Loading branch-2 data from MAT: {self.branch2_data_path}"
            )
            (
                n2,
                R2,
                self.EV0_branch2,
                self.EV_trajectories_branch2,
                _,
                _,
                self.min_size_branch2,
            ) = load_data(
                data_path=self.branch2_data_path,
                show_position=False,
                show_max_min=False,
            )

            # Consistency check
            if not np.allclose(self.n, n2):
                raise ValueError(
                    "Branch 1 and branch 2 have different n-grids; "
                    "this is not supported."
                )
            if not np.allclose(self.R, R2):
                raise ValueError(
                    "Branch 1 and branch 2 have different R-grids; "
                    "this is not supported."
                )

            self.EV_trajectories_branch2_reshaped = reshape_EV_trajectories(
                self.EV_trajectories_branch2, self.min_size_branch2
            )

            self.logger.info(
                "Branch-2 data loaded and reshaped "
                f"(min_size_branch2={self.min_size_branch2})."
            )
        else:
            self.logger.info("Single-branch configuration (num_acoustic_branches = 1).")

        self.logger.info("Data loading completed.")

    # --------------------------------------------------------------
    def prepare(self):
        """
        Convert object-typed EV trajectories into numeric complex arrays.

        For MAT-data, this yields shape [R, τ, n] after prepare_mu,
        where each entry is the (complex) acoustic eigenvalue for the
        corresponding (R, τ, n) index.

        For branch 2, if present, the same conversion is applied.
        """

        # Branch 1
        self.EV_trajectories_stacked = prepare_mu(
            self.EV_trajectories_reshaped, self.min_size
        )

        if self.use_only_acoustic:
            self.logger.info(
                "Using only acoustic part (first eigenvalue per [R, τ, n]) "
                "for branch 1 (legacy behavior)."
            )
            # No extra slicing needed; prepare_mu already takes first eigenvalue.

        # Branch 2
        if (
            self.num_acoustic_branches >= 2
            and self.EV_trajectories_branch2_reshaped is not None
        ):
            self.EV_trajectories_branch2_stacked = prepare_mu(
                self.EV_trajectories_branch2_reshaped, self.min_size_branch2
            )
            self.logger.info(
                "Branch-2 trajectories converted to numeric complex array."
            )
        else:
            self.EV_trajectories_branch2_stacked = None

    # --------------------------------------------------------------
    def find_mu_fit(self, config, tau):
        """
        First-order μ-fit (linear LSQ), multi-trajectory aware.
        """

        self.logger.info("Computing first-order μ fit (stacked trajectories)")

        EV0 = np.hstack(self.EV0).ravel()
        n = self.n
        stacked1 = self.EV_trajectories_stacked  # branch 1
        stacked2 = self.EV_trajectories_branch2_stacked  # may be None

        num_R = stacked1.shape[0]
        mu_array = np.zeros((num_R, 2))

        for i in range(num_R):
            A_blocks = []
            b_blocks = []

            # ---------------------------------
            # Reference eigenvalues
            # ---------------------------------
            s_1 = EV0[i]

            s_2 = None
            if self.EV0_branch2 is not None:
                s_2 = np.hstack(self.EV0_branch2).ravel()[i]
            else:
                try:
                    s_2 = config.w_R_table[float(self.R[i])]
                except Exception:
                    s_2 = config.w[1]

            # ---------------------------------
            # Branch 1 (always)
            # ---------------------------------
            w1 = stacked1[i, 0, :].imag
            sigma1 = stacked1[i, 0, :].real

            A1 = linfit.build_A(
                n, tau,
                w1, sigma1,
                s_ref=s_1,
                use_only_acoustic=True
            )
            b1 = linfit.build_b(
                config, n, tau,
                w1, sigma1,
                s_ref=s_1,
                use_only_acoustic=True
            )

            A_blocks.append(A1)
            b_blocks.append(b1)

            # ---------------------------------
            # Branch 2 (optional)
            # ---------------------------------
            if stacked2 is not None and 2 in self.fit_branches:

                w2 = stacked2[i, 0, :].imag
                sigma2 = stacked2[i, 0, :].real

                A2 = linfit.build_A(
                    n, tau,
                    w2, sigma2,
                    s_ref=s_2,
                    use_only_acoustic=True
                )
                b2 = linfit.build_b(
                    config, n, tau,
                    w2, sigma2,
                    s_ref=s_2,
                    use_only_acoustic=True
                )

                A_blocks.append(A2)
                b_blocks.append(b2)


            # -------------------------
            # Global stacked LSQ
            # -------------------------
            A = np.vstack(A_blocks)
            b = np.concatenate(b_blocks)

            mu = linfit.regression(
                A, b,
                check_condition_number=True,
                quiet=False
            )

            mu_array[i, :] = mu

            self.logger.info(
                f"R[{i}]={self.R[i]} | μ={mu} | cond(A)={np.linalg.cond(A):.2e}"
            )

        self.logger.info("First-order μ fit completed (stacked)")
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
        Second-order μ-fit (per R), multi-branch aware.

        For each R[i], it:
          - chooses s_1 (branch-1 EV0) and s_2 (branch-2 EV0 if
            available, otherwise from config.w_R_table or config.w[1]),
          - stacks all n-samples (and optional extra branches)
            into a global M μ - b system,
          - calls second_order_fit.solve_mu_per_R to solve a
            bounded nonlinear LSQ for p (μ-parameters).

        Returns
        -------
        mu_array : np.ndarray
            Shape (num_R, p_dim), where p_dim = 6 (symmetric) or 8.
            Each row is the μ-parameter vector p for that R.
        R : np.ndarray
            R-grid used for fitting.
        EV0_flat_1 : np.ndarray
            Flattened base eigenvalues (branch 1).
        """

        if 1 not in self.fit_branches:
            raise ValueError(
                "fit_branches must contain branch 1. "
                "Branch 1 defines the reference mode s_1."
            )

        self.logger.info("Computing second-order μ fit (per R, global, multi-branch)")

        # Flatten EV0 for branch 1
        EV0_flat_1 = np.hstack(self.EV0).ravel()
        n = self.n
        stacked1 = self.EV_trajectories_stacked  # [num_R, τ, n]
        num_R = stacked1.shape[0]

        # Optional branch-2 EV0 flattening
        EV0_flat_2 = None
        if self.EV0_branch2 is not None:
            try:
                EV0_flat_2 = np.hstack(self.EV0_branch2).ravel()
            except Exception as exc:
                self.logger.error(f"Failed to flatten EV0_branch2: {exc}")
                raise

        p_list = []

        for i in range(num_R):
            # ------------------------
            # Base modes for this R[i]
            # ------------------------
            s_1 = EV0_flat_1[i]

            # Prefer EV0_branch2 as s_2 if available
            if EV0_flat_2 is not None:
                s_2 = EV0_flat_2[i]
            else:
                R_val = float(self.R[i])
                try:
                    s_2 = config.w_R_table[R_val]
                except Exception:
                    s_2 = config.w[1]
                    self.logger.debug(
                        f"Using config.w[1] as s_2 fallback at R={R_val}."
                    )

            # Acoustic trajectory for branch 1 at this R (τ index = 0)
            w_big_1 = stacked1[i, 0, :].imag
            sigma_big_1 = stacked1[i, 0, :].real

            w_big_1 = np.asarray(w_big_1, float)
            sigma_big_1 = np.asarray(sigma_big_1, float)

            # -----------------------------------
            # Optional extra branch: branch 2
            # -----------------------------------
            extra_branches = []

            if (
                2 in self.fit_branches
                and self.num_acoustic_branches >= 2
                and self.EV_trajectories_branch2_stacked is not None
            ):
                w_big_2 = self.EV_trajectories_branch2_stacked[i, 0, :].imag
                sigma_big_2 = self.EV_trajectories_branch2_stacked[i, 0, :].real

                w_big_2 = np.asarray(w_big_2, float)
                sigma_big_2 = np.asarray(sigma_big_2, float)

                extra_branches.append(
                    {
                        "w_big": w_big_2,
                        "sigma_big": sigma_big_2,
                        "s_ref": s_2,
                        "s_other": s_1,
                    }
                )

            # -------------------------------------------------
            # Linear initialization
            # -------------------------------------------------
            mu11_init = self._linear_mu_complex(
                config=config,
                n=n,
                tau=tau,
                w_big=w_big_1,
                sigma_big=sigma_big_1,
                s_ref=s_1,
            )

            mu22_init = None
            if extra_branches:
                w_big_2 = extra_branches[0]["w_big"]
                sigma_big_2 = extra_branches[0]["sigma_big"]
                s_ref_2 = extra_branches[0]["s_ref"]

                mu22_init = self._linear_mu_complex(
                    config=config,
                    n=n,
                    tau=tau,
                    w_big=w_big_2,
                    sigma_big=sigma_big_2,
                    s_ref=s_ref_2,
                )

            def _fmt_mu(mu):
                if mu is None:
                    return "None"
                return (
                    f"{mu.real:+.4e} {mu.imag:+.4e}j "
                    f"(abs={abs(mu):.4e}, arg={np.angle(mu):+.3f})"
                )

            self.logger.info(
                f"[init μ | R-index={i} | R={self.R[i]:.3f}]\n"
                f"  mu11_init = {_fmt_mu(mu11_init)}\n"
                f"  mu22_init = {_fmt_mu(mu22_init)}"
            )


            # -------------------------------------------------
            # Second-order solve (diagonals warm-started)
            # -------------------------------------------------
            p_opt, info = sofit.solve_mu_per_R(
                config=config,
                n=n,
                tau=tau,
                w_big=w_big_1,
                sigma_big=sigma_big_1,
                s_1=s_1,
                s_2=s_2,
                use_only_acoustic=use_only_acoustic,
                enforce_symmetry=enforce_symmetry,
                weights=None,
                quiet=False,
                extra_branches=extra_branches if extra_branches else None,
                init_mu11=mu11_init,
                init_mu22=mu22_init,
            )


            p_list.append(p_opt)
            b1 = info.get("branch1_summary") or {}
            b2 = info.get("branch2_summary") or {}

            p_dim = 6 if enforce_symmetry else 8
            self.logger.info(
                "\n"
                f"[μ-fit | R-index={i} | R={self.R[i]:.3f}]\n"
                f"  Modes:\n"
                f"    s1 = {s_1}\n"
                f"    s2 = {s_2}\n"
                f"  Parameters:\n"
                f"    p = {np.array2string(p_opt, precision=3, suppress_small=True)}\n"
                f"  Residuals:\n"
                f"    cost        = {info.get('global_cost', np.nan):.3e}\n"
                f"    ||res||     = {info.get('residual_norm', np.nan):.3e}\n"
                f"  Conditioning:\n"
                f"    cond(M)     = {info.get('cond_M', np.nan):.2e}\n"
                f"    cond(M_s)   = {info.get('cond_M_scaled', np.nan):.2e}\n"
                f"  Branch-1 (scaled):\n"
                f"    rank        = [{b1.get('rank_scaled_min', np.nan)}, "
                f"{b1.get('rank_scaled_max', np.nan)}]\n"
                f"    svd_min     = {b1.get('svd_ratio_scaled_min', np.nan):.2e}\n"
                f"  Branch-2 (scaled):\n"
                f"    rank        = [{b2.get('rank_scaled_min', np.nan)}, "
                f"{b2.get('rank_scaled_max', np.nan)}]\n"
                f"    svd_min     = {b2.get('svd_ratio_scaled_min', np.nan):.2e}\n"
                f"  Meta:\n"
                f"    p_dim       = {p_dim}\n"
                f"    svd_thresh  = {info.get('mu_svd_rel_thresh', np.nan):.1e}\n"
    )

        mu_array = np.vstack(p_list)  # (num_R, p_dim)
        self.logger.info("Second-order μ fit completed (per-R global solve)")

        # Optional: build a simple global μ(R) model (e.g. linear in R)
        degree = getattr(self.config, "mu_R_poly_degree", 1)
        self.mu_R_coeffs = self.fit_mu_R_model(mu_array, degree=degree)

        return mu_array, self.R, EV0_flat_1

    def fit_mu_R_model(self, mu_array: np.ndarray, degree: int = 1):
        """
        Fit a polynomial model for μ(R) after per-R fitting.

        Parameters
        ----------
        mu_array : ndarray, shape (num_R, p_dim)
            p(R_i) values returned by the second-order fit (Re/Im).
        degree : int
            Polynomial degree for each component: p_k(R) = Σ_j c_{k,j} R^j.

        Returns
        -------
        coeffs : ndarray, shape (p_dim, degree+1)
            For each parameter index k, coeffs[k] contains the polynomial
            coefficients in descending powers of R (np.polyfit convention).
        """

        R = np.asarray(self.R, float)
        num_R, p_dim = mu_array.shape

        coeffs = np.zeros((p_dim, degree + 1), dtype=float)

        for k in range(p_dim):
            y = mu_array[:, k]
            # ignore NaNs if any
            mask = np.isfinite(y)
            if np.count_nonzero(mask) <= degree:
                # not enough points, fall back to simple mean
                coeffs[k, :] = 0.0
                coeffs[k, -1] = np.nanmean(y[mask]) if np.any(mask) else 0.0
            else:
                coeffs[k, :] = np.polyfit(R[mask], y[mask], degree)

        self.logger.info(
            f"Fitted μ(R) polynomial model of degree {degree} for {p_dim} parameters."
        )
        return coeffs

    def _linear_mu_complex(self, config, n, tau, w_big, sigma_big, s_ref):
        """
        Run linear μ-fit for a single branch and return complex μ.
        """
        A = linfit.build_A(
            n, tau,
            w_big, sigma_big,
            s_ref=s_ref,
            use_only_acoustic=True
        )

        b = linfit.build_b(
            config, n, tau,
            w_big, sigma_big,
            s_ref=s_ref,
            use_only_acoustic=True
        )

        mu_re_im = linfit.regression(
            A, b,
            check_condition_number=True,
            quiet=True
        )

        return mu_re_im[0] + 1j * mu_re_im[1]
