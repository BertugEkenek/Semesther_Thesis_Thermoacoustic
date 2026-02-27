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
        data_path: str | None,
        txt_solution_path: str,
        order: int,
        use_txt_solutions: bool,
        enforce_symmetry: bool,
        num_acoustic_branches: int = 1,
        fit_branches=None,
        branch2_data_path: str | None = None,
        merged_optimization: bool = False,
        data_paths_map: dict | None = None,
    ):
        self.data_path = data_path
        self.txt_solution_path = txt_solution_path
        self.order = order
        self.config = config
        self.logger = logger
        self.use_txt_solutions = use_txt_solutions
        self.enforce_symmetry = enforce_symmetry

        self.num_acoustic_branches = int(num_acoustic_branches)
        self.fit_branches = fit_branches if fit_branches is not None else [1]
        self.branch2_data_path = branch2_data_path

        self.merged_optimization = merged_optimization
        self.data_paths_map = data_paths_map
        self.data_store = {}  # Stores data for multiple taus: {tau: { ... }}

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
        if self.merged_optimization:
            self._load_merged_data()
            # Perform reshape for all loaded datasets
            for tau, data in self.data_store.items():
                data['EV_trajectories_reshaped'] = reshape_EV_trajectories(
                    data['EV_trajectories'], data['min_size']
                )
                if self.num_acoustic_branches >= 2:
                    data['EV_trajectories_branch2_reshaped'] = reshape_EV_trajectories(
                        data['EV_trajectories_branch2'], data['min_size_branch2']
                    )
            self.logger.info("Merged data loading and reshaping completed.")
            return

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
        if self.data_path is None:
            raise ValueError("data_path is None, but _load_mat was called (merged_optimization=False).")

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

    def _load_merged_data(self):
        if not self.data_paths_map:
            raise ValueError("merged_optimization is True but data_paths_map is missing.")

        first = True
        for tau, paths in self.data_paths_map.items():
            self.logger.info(f"Loading merged data for tau={tau}")
            
            # Load branch 1
            (n, R, EV0, EV_traj, _, _, min_size) = load_data(
                data_path=paths['branch1'], show_position=False, show_max_min=False
            )

            # Load branch 2 if needed
            EV0_2, EV_traj_2, min_size_2 = None, None, None
            if self.num_acoustic_branches >= 2:
                (n2, R2, EV0_2, EV_traj_2, _, _, min_size_2) = load_data(
                    data_path=paths['branch2'], show_position=False, show_max_min=False
                )
                if not np.allclose(n, n2) or not np.allclose(R, R2):
                    raise ValueError(f"Mismatch between branch 1 and 2 for tau={tau}")

            # Validate consistency across taus (R and n must be identical)
            if first:
                self.n = n
                self.R = R
                # Set fallback EV0/EV0_branch2 for orchestrator compatibility
                # (Orchestrator might access these attributes directly)
                self.EV0 = EV0
                self.EV0_branch2 = EV0_2
                self.min_size = min_size
                self.min_size_branch2 = min_size_2
                first = False
            else:
                if not np.allclose(self.n, n):
                    raise ValueError(f"n grid mismatch for tau={tau} vs previous.")
                if not np.allclose(self.R, R):
                    raise ValueError(f"R grid mismatch for tau={tau} vs previous.")

            self.data_store[tau] = {
                'n': n, 'R': R, 'EV0': EV0, 'EV_trajectories': EV_traj, 'min_size': min_size,
                'EV0_branch2': EV0_2, 'EV_trajectories_branch2': EV_traj_2, 'min_size_branch2': min_size_2
            }

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
        if self.merged_optimization:
            for tau, data in self.data_store.items():
                data['EV_trajectories_stacked'] = prepare_mu(
                    data['EV_trajectories_reshaped'], data['min_size']
                )
                if self.num_acoustic_branches >= 2 and data['EV_trajectories_branch2_reshaped'] is not None:
                    data['EV_trajectories_branch2_stacked'] = prepare_mu(
                        data['EV_trajectories_branch2_reshaped'], data['min_size_branch2']
                    )
            self.logger.info("Merged data preparation (numeric conversion) completed.")
            return

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

        if self.merged_optimization:
            return self._find_mu_fit_merged_linear(config, tau, branch_id)

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

            A = linfit.build_A(n, tau, w, sigma, s_ref=s_ref)

            cfg = config.get_branch_config(branch_id)
            b = linfit.build_b(cfg, n, tau, w, sigma, s_ref=s_ref)

            mu = linfit.regression(A, b, check_condition_number=True, quiet=False)
            mu_array[i, :] = mu

            self.logger.info(
                f"[Linear μ | branch={branch_id}] R[{i}]={self.R[i]} | μ={mu} | cond(A)={np.linalg.cond(A):.2e}"
            )
        logger.info(f"Branch {branch_id}: ν/α = {cfg.nu[0] / cfg.alpha[0]:.2e}")
        self.logger.info(f"First-order μ fit completed (single-branch={branch_id})")
        return mu_array, self.R, EV0_flat

    def _find_mu_fit_merged_linear(self, config, target_tau, branch_id):
        """
        Merged First-order μ-fit.
        Optimizes one μ across ALL datasets in self.data_store.
        Returns EV0 corresponding to 'target_tau' for plotting consistency.
        """
        # 1. Prepare output array
        num_R = len(self.R)
        mu_array = np.zeros((num_R, 2))

        # 2. Iterate per R (global optimization at each R)
        for i in range(num_R):
            A_rows = []
            b_rows = []

            # 3. Collect data from all training taus
            for t_val, data in self.data_store.items():
                # Select branch data
                if branch_id == 1:
                    stacked = data['EV_trajectories_stacked']
                    EV0_flat = np.hstack(data['EV0']).ravel()
                else:
                    stacked = data['EV_trajectories_branch2_stacked']
                    EV0_flat = np.hstack(data['EV0_branch2']).ravel()

                s_ref = EV0_flat[i]
                w = stacked[i, 0, :].imag
                sigma = stacked[i, 0, :].real

                # Build matrices for this specific tau
                A_i = linfit.build_A(self.n, t_val, w, sigma, s_ref=s_ref)
                cfg = config.get_branch_config(branch_id)
                b_i = linfit.build_b(cfg, self.n, t_val, w, sigma, s_ref=s_ref)

                A_rows.append(A_i)
                b_rows.append(b_i)

            # 4. Stack and Solve
            A_total = np.vstack(A_rows)
            b_total = np.concatenate(b_rows)
            mu = linfit.regression(A_total, b_total, check_condition_number=True, quiet=True)
            mu_array[i, :] = mu

        self.logger.info(f"Merged First-order μ fit completed (branch={branch_id}, {len(self.data_store)} datasets)")
        
        # 5. Retrieve EV0 for the target tau (for plotting)
        if target_tau in self.data_store:
            data_target = self.data_store[target_tau]
            if branch_id == 1:
                EV0_target = np.hstack(data_target['EV0']).ravel()
            else:
                EV0_target = np.hstack(data_target['EV0_branch2']).ravel()
        else:
            self.logger.warning(f"Target tau {target_tau} not in training set. Returning None for EV0.")
            EV0_target = None

        return mu_array, self.R, EV0_target

    def _linear_mu_complex(self, config, branch_id, n, tau, w_big, sigma_big, s_ref):
        w_big = np.asarray(w_big).reshape(-1)
        sigma_big = np.asarray(sigma_big).reshape(-1)

        cfg = config.get_branch_config(branch_id)
        A = linfit.build_A(n, tau, w_big, sigma_big, s_ref=s_ref)
        b = linfit.build_b(cfg, n, tau, w_big, sigma_big, s_ref=s_ref)

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
        enforce_symmetry: bool,
    ):
        if 1 not in self.fit_branches:
            raise ValueError("fit_branches must contain branch 1 (defines reference mode s_1).")

        self.logger.info("Computing second-order μ fit (per R, global, multi-branch)")

        cond_list = []
        rank_list = []
        
        n = self.n
        num_R = len(self.R)

        p_list = []
        prev_p = None
        for i in range(num_R):
            # Optional: use linear init (branch-aware)
            init_mu11 = None
            init_mu22 = None

            # --- μ11 init (branch 1) ---
            try:
                if self.merged_optimization and tau in self.data_store:
                    d_init = self.data_store[tau]
                    st1_init = d_init['EV_trajectories_stacked']
                    ev1_init = np.hstack(d_init['EV0']).ravel()
                    w_init = st1_init[i, 0, :].imag
                    sig_init = st1_init[i, 0, :].real
                    s1_init = ev1_init[i]
                else:
                    st1_init = self._stacked(1)
                    ev1_init = self._ev0_flat(1)
                    w_init = st1_init[i, 0, :].imag
                    sig_init = st1_init[i, 0, :].real
                    s1_init = ev1_init[i]

                mu11_lin = self._linear_mu_complex(config, 1, n, tau, w_init, sig_init, s_ref=s1_init)
                init_mu11 = self._validate_linear_mu_init(mu11_lin, 1, float(self.R[i]))
            except Exception:
                init_mu11 = None


            # --- μ22 init (branch 2) [NEW] ---
            try:
                # only if branch-2 data exists
                if self.num_acoustic_branches >= 2 and (2 in self.fit_branches):

                    if self.merged_optimization and tau in self.data_store:
                        d_init = self.data_store[tau]
                        st2_init = d_init.get('EV_trajectories_branch2_stacked')
                        ev2_obj = d_init.get('EV0_branch2')
                        if st2_init is None or ev2_obj is None:
                            raise ValueError("Merged branch-2 data missing for init_mu22.")
                        ev2_init = np.hstack(ev2_obj).ravel()

                        w2_init = st2_init[i, 0, :].imag
                        sig2_init = st2_init[i, 0, :].real
                        s2_init = ev2_init[i]

                    else:
                        st2_init = self._stacked(2)
                        ev2_init = self._ev0_flat(2)
                        w2_init = st2_init[i, 0, :].imag
                        sig2_init = st2_init[i, 0, :].real
                        s2_init = ev2_init[i]

                    mu22_lin = self._linear_mu_complex(config, 2, n, tau, w2_init, sig2_init, s_ref=s2_init)
                    init_mu22 = self._validate_linear_mu_init(mu22_lin, 2, float(self.R[i]))

            except Exception:
                init_mu22 = None


            # ------------------------------------------------------
            # Build Data Blocks
            # ------------------------------------------------------
            data_blocks = []

            def add_blocks(t_val, st1, ev1, st2, ev2, R_val):
                # s_1 for this condition
                s_1_local = ev1[i]
                
                # s_2 for this condition
                if ev2 is not None:
                    s_2_local = ev2[i]
                else:
                    try:
                        s_2_local = config.w_R_table[float(R_val)]
                    except Exception:
                        s_2_local = config.w[1]

                # Branch 1
                w1 = st1[i, 0, :].imag
                sig1 = st1[i, 0, :].real
                data_blocks.append({
                    'tag': f'tau={t_val:.4f}_b1',
                    'tau': t_val,
                    'w': w1,
                    'sigma': sig1,
                    's_ref': s_1_local,
                    's_1': s_1_local,
                    's_2': s_2_local
                })

                # Branch 2
                if st2 is not None and (2 in self.fit_branches):
                    w2 = st2[i, 0, :].imag
                    sig2 = st2[i, 0, :].real
                    data_blocks.append({
                        'tag': f'tau={t_val:.4f}_b2',
                        'tau': t_val,
                        'w': w2,
                        'sigma': sig2,
                        's_ref': s_2_local,
                        's_1': s_1_local,
                        's_2': s_2_local
                    })

            if self.merged_optimization:
                for t_val, d in self.data_store.items():
                    st1 = d['EV_trajectories_stacked']
                    ev1 = np.hstack(d['EV0']).ravel()
                    st2 = d.get('EV_trajectories_branch2_stacked')
                    ev2 = None
                    if st2 is not None:
                        ev2 = np.hstack(d['EV0_branch2']).ravel()
                    add_blocks(t_val, st1, ev1, st2, ev2, self.R[i])
            else:
                st1 = self._stacked(1)
                ev1 = self._ev0_flat(1)
                st2 = None
                ev2 = None
                if self.num_acoustic_branches >= 2:
                    st2 = self._stacked(2)
                    ev2 = self._ev0_flat(2)
                add_blocks(tau, st1, ev1, st2, ev2, self.R[i])
            
            # Solve per-R global nonlinear LSQ
            p_opt, info = sofit.solve_mu_per_R(
                config=config,
                n=n,
                data_blocks=data_blocks,
                enforce_symmetry=enforce_symmetry,
                quiet=True,
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

        # Return EV0 for the target tau (for plotting consistency)
        if self.merged_optimization and tau in self.data_store:
            EV0_ret = np.hstack(self.data_store[tau]['EV0']).ravel()
        else:
            EV0_ret = self._ev0_flat(1)

        return mu_array, self.R, EV0_ret
