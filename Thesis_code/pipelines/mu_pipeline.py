# pipelines/mu_pipeline.py

import numpy as np

from utils import logger
from dataio.dataloader import load_data, reshape_EV_trajectories, load_txt_solutions
from pipelines.prep_work import prepare_mu
from fitting import linear_fit as linfit
from fitting import second_order_fit as sofit


class MUFITPipeline:
    """
    Pipeline for:
      - loading eigenvalue data (MAT or TXT) for acoustic branch 1
      - optionally loading a second dataset for acoustic branch 2
      - reshaping + converting EV trajectories to numeric form
      - fitting μ (first-order linear; second-order global per-R nonlinear)

    IMPORTANT CONTRACT (current project):
      After prepare_mu, MAT trajectories become numeric complex arrays
      with shape [num_R, τ, n]. Each entry is ONE complex eigenvalue
      (prepare_mu takes the first element if the raw entry was an array).
      See prepare_mu implementation.  :contentReference[oaicite:4]{index=4}
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
        num_acoustic_branches: int = 1,
        fit_branches=None,
        branch2_data_path: str | None = None,
    ):
        self.data_path = data_path
        self.txt_solution_path = txt_solution_path
        self.order = order
        self.config = config
        self.logger = logger
        self.use_only_acoustic = use_only_acoustic
        self.use_txt_solutions = use_txt_solutions
        self.enforce_symmetry = enforce_symmetry

        self.num_acoustic_branches = int(num_acoustic_branches)
        self.fit_branches = fit_branches if fit_branches is not None else [1]
        self.branch2_data_path = branch2_data_path

        # branch 1
        self.n = None
        self.R = None
        self.EV0 = None
        self.EV_trajectories = None
        self.min_size = None
        self.EV_trajectories_reshaped = None
        self.EV_trajectories_stacked = None  # numeric [R, τ, n]

        # branch 2 (optional)
        self.EV0_branch2 = None
        self.EV_trajectories_branch2 = None
        self.min_size_branch2 = None
        self.EV_trajectories_branch2_reshaped = None
        self.EV_trajectories_branch2_stacked = None  # numeric [R, τ, n]

        # optional μ(R) model
        self.mu_R_coeffs = None

    # ==========================================================
    # loading
    # ==========================================================
    def load_all_data(self):
        if self.use_txt_solutions:
            self._load_txt()
        else:
            self._load_mat()

        # shared reshape
        self.EV_trajectories_reshaped = reshape_EV_trajectories(
            self.EV_trajectories, self.min_size
        )
        if self.num_acoustic_branches >= 2:
            self.EV_trajectories_branch2_reshaped = reshape_EV_trajectories(
                self.EV_trajectories_branch2, self.min_size_branch2
            )

        self.logger.info("Data loading completed.")

    def _load_txt(self):
        if self.num_acoustic_branches == 1:
            (self.n, self.R, self.EV0, self.EV_trajectories, _, _, self.min_size) = \
                load_txt_solutions(self.txt_solution_path + "_branch1.txt")
            self.EV0_branch2 = None
            self.EV_trajectories_branch2 = None
            self.min_size_branch2 = None
            return

        if self.num_acoustic_branches == 2:
            (self.n, self.R, self.EV0, self.EV_trajectories, _, _, self.min_size) = \
                load_txt_solutions(self.txt_solution_path + "_branch1.txt")

            (n2, R2, self.EV0_branch2, self.EV_trajectories_branch2, _, _, self.min_size_branch2) = \
                load_txt_solutions(self.txt_solution_path + "_branch2.txt")

            if not np.allclose(self.n, n2):
                raise ValueError("TXT branch 1 and branch 2 have different n grids.")
            if not np.allclose(self.R, R2):
                raise ValueError("TXT branch 1 and branch 2 have different R grids.")
            return

        raise ValueError("TXT loader supports only 1 or 2 acoustic branches")

    def _load_mat(self):
        self.logger.info(f"Loading branch-1 data from MAT: {self.data_path}")
        (self.n, self.R, self.EV0, self.EV_trajectories, _, _, self.min_size) = load_data(
            data_path=self.data_path,
            show_position=False,
            show_max_min=False,
        )

        if self.num_acoustic_branches < 2:
            return

        if self.branch2_data_path is None:
            raise ValueError("num_acoustic_branches >= 2, but branch2_data_path is not set.")

        self.logger.info(f"Loading branch-2 data from MAT: {self.branch2_data_path}")
        (n2, R2, self.EV0_branch2, self.EV_trajectories_branch2, _, _, self.min_size_branch2) = load_data(
            data_path=self.branch2_data_path,
            show_position=False,
            show_max_min=False,
        )

        if not np.allclose(self.n, n2):
            raise ValueError("Branch 1 and branch 2 have different n grids.")
        if not np.allclose(self.R, R2):
            raise ValueError("Branch 1 and branch 2 have different R grids.")

    # ==========================================================
    # preparation
    # ==========================================================
    def prepare(self):
        """
        Convert object-typed EV trajectories into numeric complex arrays.

        Current project behavior:
          - MAT becomes [R, τ, n] complex after prepare_mu
          - each entry is the first eigenvalue in the stored array (if any)
        """
        self.EV_trajectories_stacked = prepare_mu(
            self.EV_trajectories_reshaped, self.min_size
        )

        if self.num_acoustic_branches >= 2 and self.EV_trajectories_branch2_reshaped is not None:
            self.EV_trajectories_branch2_stacked = prepare_mu(
                self.EV_trajectories_branch2_reshaped, self.min_size_branch2
            )
            self.logger.info("Branch-2 trajectories converted to numeric complex array.")
        else:
            self.EV_trajectories_branch2_stacked = None

    # ==========================================================
    # helpers
    # ==========================================================
    def _ev0_flat(self, branch_id: int):
        if branch_id == 1:
            return np.hstack(self.EV0).ravel()
        if branch_id == 2:
            if self.EV0_branch2 is None:
                return None
            return np.hstack(self.EV0_branch2).ravel()
        raise ValueError(f"branch_id must be 1 or 2, got {branch_id}")

    def _stacked(self, branch_id: int):
        if branch_id == 1:
            if self.EV_trajectories_stacked is None:
                raise ValueError("Branch-1 trajectories not prepared. Call load_all_data()+prepare() first.")
            return self.EV_trajectories_stacked
        if branch_id == 2:
            if self.num_acoustic_branches < 2 or self.EV_trajectories_branch2_stacked is None:
                raise ValueError("Branch-2 data not available/prepared.")
            return self.EV_trajectories_branch2_stacked
        raise ValueError(f"branch_id must be 1 or 2, got {branch_id}")

    # ==========================================================
    # linear μ-fit (single branch; legacy)
    # ==========================================================
    def find_mu_fit(self, config, tau, branch_id: int = 1):
        """
        First-order μ-fit (linear LSQ), SINGLE-BRANCH ONLY.
        Uses τ-index 0, all n samples.
        """
        if branch_id not in (1, 2):
            raise ValueError(f"branch_id must be 1 or 2, got {branch_id}")

        EV0_flat = self._ev0_flat(branch_id)
        stacked = self._stacked(branch_id)

        n = self.n
        num_R = stacked.shape[0]
        mu_array = np.zeros((num_R, 2))

        for i in range(num_R):
            s_ref = EV0_flat[i]

            # core contract: stacked[i, 0, :] is length n
            w = stacked[i, 0, :].imag
            sigma = stacked[i, 0, :].real

            A = linfit.build_A(n, tau, w, sigma, s_ref=s_ref, use_only_acoustic=True)

            cfg = config.get_branch_config(branch_id)
            b = linfit.build_b(cfg, n, tau, w, sigma, s_ref=s_ref, use_only_acoustic=True)

            mu = linfit.regression(A, b, check_condition_number=True, quiet=False)
            mu_array[i, :] = mu

            self.logger.info(
                f"[Linear μ | branch={branch_id}] R[{i}]={self.R[i]} | μ={mu} | cond(A)={np.linalg.cond(A):.2e}"
            )

        self.logger.info(f"First-order μ fit completed (single-branch={branch_id})")
        return mu_array, self.R, EV0_flat

    def _linear_mu_complex(self, config, branch_id, n, tau, w_big, sigma_big, s_ref):
        w_big = np.asarray(w_big).reshape(-1)
        sigma_big = np.asarray(sigma_big).reshape(-1)

        cfg = config.get_branch_config(branch_id)
        A = linfit.build_A(n, tau, w_big, sigma_big, s_ref=s_ref, use_only_acoustic=True)
        b = linfit.build_b(cfg, n, tau, w_big, sigma_big, s_ref=s_ref, use_only_acoustic=True)

        mu_re_im = linfit.regression(A, b, check_condition_number=True, quiet=True)
        return mu_re_im[0] + 1j * mu_re_im[1]

    def find_mu_fit_linear_selected_branches(self, config, tau):
        """
        Run linear μ-fit for each branch listed in self.fit_branches.
        """
        results = {}
        for branch_id in self.fit_branches:
            if branch_id not in (1, 2):
                raise ValueError(f"Invalid branch_id={branch_id} in fit_branches")

            mu_array, R, EV0_flat = self.find_mu_fit(config=config, tau=tau, branch_id=branch_id)
            results[branch_id] = {"mu": mu_array, "R": R, "EV0": EV0_flat}
        return results

    def _validate_linear_mu_init(self, mu, branch_id, R):
        """
        If Re(mu) < 0:
          - branch 2: force 1+0j (so μ22 init is neutral)
          - branch 1: drop initializer (None)
        """
        if mu is None:
            return None

        if np.real(mu) < 0:
            self.logger.warning(
                f"[μ init rejected] branch={branch_id}, R={R:.3f} | "
                f"Re(μ{branch_id}{branch_id})={mu.real:.3e} < 0 → using fallback"
            )
            return (1.0 + 0.0j) if branch_id == 2 else None

        return mu

    # ==========================================================
    # second-order μ-fit (global per-R; multi-branch aware)
    # ==========================================================
    def find_mu_fit_second_order(
        self,
        config: object,
        tau: float,
        order: int,
        use_only_acoustic: bool,
        enforce_symmetry: bool,
    ):
        if 1 not in self.fit_branches:
            raise ValueError("fit_branches must contain branch 1 (defines reference mode s_1).")

        self.logger.info("Computing second-order μ fit (per R, global, multi-branch)")

        cond_list = []
        rank_list = []
        
        n = self.n
        stacked1 = self._stacked(1)  # [num_R, τ, n]
        EV0_flat_1 = self._ev0_flat(1)
        num_R = stacked1.shape[0]

        EV0_flat_2 = self._ev0_flat(2)  # may be None

        p_list = []
        prev_p = None
        for i in range(num_R):
            # Base reference eigenvalues for this R
            s_1 = EV0_flat_1[i]

            if EV0_flat_2 is not None:
                s_2 = EV0_flat_2[i]
            else:
                R_val = float(self.R[i])
                try:
                    s_2 = config.w_R_table[R_val]
                except Exception:
                    s_2 = config.w[1]

            # branch 1: τ index 0, all n samples
            w_big_1 = stacked1[i, 0, :].imag
            sigma_big_1 = stacked1[i, 0, :].real

            # Optional: use linear init (branch-aware)
            init_mu11 = None
            init_mu22 = None
            try:
                mu11_lin = self._linear_mu_complex(config, 1, n, tau, w_big_1, sigma_big_1, s_ref=s_1)
                init_mu11 = self._validate_linear_mu_init(mu11_lin, 1, float(self.R[i]))
            except Exception:
                init_mu11 = None

            # extra branches (constraints)
            extra = []
            if self.num_acoustic_branches >= 2 and (2 in self.fit_branches):
                stacked2 = self._stacked(2)
                w_big_2 = stacked2[i, 0, :].imag
                sigma_big_2 = stacked2[i, 0, :].real

                try:
                    mu22_lin = self._linear_mu_complex(config, 2, n, tau, w_big_2, sigma_big_2, s_ref=s_2)
                    init_mu22 = self._validate_linear_mu_init(mu22_lin, 2, float(self.R[i]))
                    
                except Exception:
                    init_mu22 = None
                
                extra.append(dict(
                    tag="branch2",
                    w_big=w_big_2,
                    sigma_big=sigma_big_2,
                    s_ref=s_2,
                ))
            
            # Solve per-R global nonlinear LSQ
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
                quiet=True,
                extra_branches=extra if extra else None,
                init_mu11=init_mu11,
                init_mu22=init_mu22,
                prev_p_opt=prev_p,    
            )

            prev_p = p_opt            
            p_list.append(p_opt)

            self.logger.info(
                f"[Second-order μ] R[{i}]={self.R[i]:.2e} | "
                f"cost={info.get('global_cost', np.nan):.3e} | "
                f"phys_cost={info.get('phys_cost', np.nan):.3e} | "
                f"phys_rms={info.get('phys_rms', np.nan):.3e} | "
                f"cond(A)={info.get('cond_A_phys', np.nan):.2e} | "
                f"rank={info.get('rank_A_phys', None)}"
            )


        mu_array = np.asarray(p_list, dtype=float)
        self.mu_cond_numbers = np.asarray(cond_list)
        self.mu_ranks = np.asarray(rank_list)


        return mu_array, self.R, EV0_flat_1

