# pipelines/orchestrator.py

import os
import numpy as np

from utils import logger
from pipelines.prep_work import initialize_plot

from pipelines.plotting_pipeline import (
    plot_eigenvalue_trajectories,
    overlay_data_if_requested,
    plot_mu_global_diagnostics,
)

from pipelines.simulation import (
    compute_reference_solutions_two_branches,
)

from pipelines.io_pipeline import (
    save_reference_and_noisy_solutions_two_branches,
)

from compare_nu_alphaK import (
    load_nu_from_mat,
    compute_reference_alphaK_nu,
    compute_fitted_alphaK_mu_nu,
    plot_alphaK_comparison_complex_plane_separate,
)


class SolveEigenWorkflow:
    """
    High-level orchestrator for:
      - μ fitting (via MUFITPipeline provided from outside)
      - solving eigenvalues
      - saving TXT reference/noisy data
      - plotting and overlays
      - optional MAT-based μ construction via μ = ν_mat / ν_config
      - comparing αKν (.mat) with αKμν (fit) in the complex plane
    """

    def __init__(self, config, mu_pipeline, log=logger):
        self.config = config
        self.mu_pipeline = mu_pipeline
        self.logger = log

    # ----------------------------------------------------------
    def _compute_mu(
        self,
        correction: bool,
        mu_order: str,
        flame_model_approximator: str,
        tau_train_list: list[float],
        save_mu: bool,
        use_saved_mu: bool,
    ):
        if not tau_train_list:
            raise ValueError("tau_train_list must be non-empty.")

        if len(tau_train_list) == 1:
            tau_tag = f"{int(round(tau_train_list[0] * 1000))}ms"
        else:
            tau_tag = "train_" + "_".join(
                f"{int(round(t * 1000))}ms" for t in sorted(float(t) for t in tau_train_list)
            )

        EV0_flat = None

        if not correction:
            mu_array = 1.0
            R = self.mu_pipeline.R
            self.logger.info("No correction: using μ = 1.0")
            return mu_array, R

        if not use_saved_mu:
            if mu_order == "First":
                lin_results = self.mu_pipeline.find_mu_fit_linear_selected_branches(
                    self.config, tau_train_list
                )

                if len(lin_results) == 1:
                    branch_id = next(iter(lin_results.keys()))
                    res = lin_results[branch_id]
                    mu_array = res["mu"]
                    R = res["R"]
                    self.logger.info(f"Linear μ-fit completed for branch {branch_id} only.")
                else:
                    mu_array = lin_results
                    R = next(iter(lin_results.values()))["R"]
                    self.logger.info(
                        f"Linear μ-fit completed for branches {list(lin_results.keys())}."
                    )

            elif mu_order == "Second":
                if not hasattr(self.config, "mu_fit_strategy"):
                    raise AttributeError(
                        "Second-order μ-fit requires config.mu_fit_strategy "
                        "(e.g. 'rank1_sym', 'sym_only', 'none', 'rank1_mag_phasefree')."
                    )

                mu_array, R, EV0_flat = self.mu_pipeline.find_mu_fit_second_order(
                    self.config, tau_train_list
                )
            else:
                raise ValueError(f"Invalid mu_order: {mu_order}")

            if save_mu:
                self._save_mu_values(mu_array, R, EV0_flat, flame_model_approximator, tau_tag)

        else:
            mu_array, R, EV0_flat = self._load_mu_values(flame_model_approximator, tau_tag)

        return mu_array, R

    # ----------------------------------------------------------
    def _match_R_from_available(self, R_value, available_R, atol=1e-10):
        available_R = np.asarray(available_R, dtype=float)
        idx = np.where(np.isclose(available_R, float(R_value), atol=atol, rtol=0.0))[0]

        if len(idx) == 0:
            raise KeyError(
                f"R={R_value} not found within atol={atol}. "
                f"Available: {available_R.tolist()}"
            )

        if len(idx) > 1:
            raise ValueError(
                f"Ambiguous R match for R={R_value}. "
                f"Matches: {available_R[idx].tolist()}"
            )

        return float(available_R[idx[0]])

    # ----------------------------------------------------------
    def _build_mu_array_from_mat_nu(self, mat_path: str, R: np.ndarray, atol: float = 1e-10):
        """
        Construct μ-array directly from MAT ν values via

            μ_ij(R) = ν_ij^(mat)(R) / ν_ij^(config)

        Returns an (nR, 8) real-valued array with columns:
            [Re11, Im11, Re22, Im22, Re12, Im12, Re21, Im21]
        """
        if not hasattr(self.config, "nu"):
            raise AttributeError("config.nu is required to build μ from MAT ν data.")

        nu_cfg = np.asarray(self.config.nu, dtype=complex).ravel()
        if nu_cfg.size < 4:
            raise ValueError(
                f"config.nu must contain at least 4 entries (11,22,12,21); got {nu_cfg.size}."
            )

        mat_nu = load_nu_from_mat(mat_path, self.config.name)
        available_R = np.array(sorted(mat_nu.keys()), dtype=float)

        mu_rows = []

        for R_val in np.asarray(R, dtype=float):
            R_match = self._match_R_from_available(R_val, available_R, atol=atol)
            nuR = mat_nu[R_match]

            mu11 = nuR["nu11"] / nu_cfg[0]
            mu22 = nuR["nu22"] / nu_cfg[1]
            mu12 = nuR["nu12"] / nu_cfg[2]
            mu21 = nuR["nu21"] / nu_cfg[3]

            mu_rows.append([
                mu11.real, mu11.imag,
                mu22.real, mu22.imag,
                mu12.real, mu12.imag,
                mu21.real, mu21.imag,
            ])

        mu_array = np.asarray(mu_rows, dtype=float)

        self.logger.info("Constructed μ-array from MAT ν data using μ = ν_mat / ν_config.")
        return mu_array

    # ----------------------------------------------------------
    def _save_mu_values(self, mu_array, R, EV0_flat, flame_model_approximator, tau_tag: str):
        save_dir = "./Results/Mu_values"
        os.makedirs(save_dir, exist_ok=True)

        base = f"mu_values_{flame_model_approximator}_{self.config.name}_{tau_tag}"

        np.savetxt(
            os.path.join(save_dir, base),
            mu_array.astype(str),
            fmt="%s",
            delimiter=",",
        )

        ev0_path = os.path.join("./data", tau_tag)
        os.makedirs(ev0_path, exist_ok=True)

        if EV0_flat is not None:
            np.savetxt(
                os.path.join(ev0_path, f"{self.config.name}_EV0_values"),
                EV0_flat.astype(str),
                fmt="%s",
                delimiter=",",
            )

        np.savetxt(
            os.path.join(ev0_path, f"{self.config.name}_R_values"),
            R,
            fmt="%.12e",
            delimiter=",",
        )

        self.logger.info(f"Saved μ-array, EV0 (if available), and R (tag={tau_tag}).")

    def _load_mu_values(self, flame_model_approximator, tau_tag: str):
        base = os.path.join(
            "./Results/Mu_values",
            f"mu_values_{flame_model_approximator}_{self.config.name}_{tau_tag}",
        )

        mu_array = np.loadtxt(base, dtype=complex, delimiter=",")

        ev0_path = os.path.join("./data", tau_tag)

        ev0_file = os.path.join(ev0_path, f"{self.config.name}_EV0_values")
        if os.path.exists(ev0_file):
            EV0_flat = np.loadtxt(ev0_file, dtype=complex, delimiter=",")
        else:
            EV0_flat = None

        R = np.loadtxt(
            os.path.join(ev0_path, f"{self.config.name}_R_values"),
            dtype=float,
            delimiter=",",
        )

        self.logger.info(f"Loaded μ-array, EV0 (if available), and R from disk (tag={tau_tag}).")
        return mu_array, R, EV0_flat

    # ----------------------------------------------------------
    def run(
        self,
        correction: bool,
        order: int,
        mu_order: str,
        F_model,
        tau_plot: float,
        tau_train_list: list[float],
        window: int,
        R_value: float,
        n_values: np.ndarray,
        Galerkin: str,
        show_tax: bool,
        save_fig: bool,
        show_fig: bool,
        filename: str,
        save_mu: bool,
        use_saved_mu: bool,
        save_solution: bool,
        use_txt_solutions: bool,
        nprandomsigma: float,
        comparison: bool,
    ):
        # ----------------------------------------------------------
        # LOCAL EXPERIMENT FLAG:
        # Use μ(R) = ν_mat(R) / ν_config instead of fitted μ
        # ----------------------------------------------------------
        USE_MAT_NU_RATIO_AS_MU = False
        MAT_NU_PATH = "./data/SimulationResults.mat"

        # ----------------------------------------------------------
        # Determine branch_id for plotting (linear case only)
        # ----------------------------------------------------------
        plot_branch_id = None
        if mu_order == "First":
            branches = self.mu_pipeline.fit_branches
            if len(branches) != 1:
                raise ValueError(
                    "Linear μ plotting requires exactly one branch. "
                    f"fit_branches={branches} is ambiguous."
                )
            plot_branch_id = branches[0]
            self.logger.info(f"Linear μ plotting will use acoustic branch {plot_branch_id}.")

        # ----------------------------------------------------------
        # 1) Compute μ
        # ----------------------------------------------------------
        if correction and USE_MAT_NU_RATIO_AS_MU:
            if mu_order != "Second":
                raise ValueError(
                    "USE_MAT_NU_RATIO_AS_MU currently supports only mu_order='Second'."
                )

            R = np.asarray(self.mu_pipeline.R, dtype=float)
            mu_array = self._build_mu_array_from_mat_nu(
                mat_path=MAT_NU_PATH,
                R=R,
                atol=1e-10,
            )

            self.logger.info("Bypassing μ fit: using μ-array built from ν_mat / ν_config.")

        else:
            mu_array, R = self._compute_mu(
                correction=correction,
                mu_order=mu_order,
                flame_model_approximator=F_model.__name__,
                tau_train_list=tau_train_list,
                save_mu=save_mu,
                use_saved_mu=use_saved_mu,
            )

        # ----------------------------------------------------------
        # αKν (.mat) vs αKμν (fit) comparison data
        # ----------------------------------------------------------
        alphaK_ref_eff = None
        alphaK_fit_eff = None

        if correction:
            try:
                mat_nu = load_nu_from_mat(MAT_NU_PATH, self.config.name)

                alphaK_ref_eff = compute_reference_alphaK_nu(
                    self.config,
                    R,
                    mat_nu,
                )

                alphaK_fit_eff = compute_fitted_alphaK_mu_nu(
                    config=self.config,
                    mu_array=mu_array,
                )

                self.logger.info(
                    "Computed αKν (.mat) and αKμν (fit) comparison data successfully."
                )

            except Exception as e:
                self.logger.warning(
                    f"Could not compute αKν vs αKμν comparison data: {e}"
                )

        # ----------------------------------------------------------
        # Plot-time EV0 must come from tau_plot dataset
        # ----------------------------------------------------------
        tau_key_plot = (
            int(round(tau_plot * 1000))
            if int(round(tau_plot * 1000)) in self.mu_pipeline.data_store
            else next(iter(self.mu_pipeline.data_store.keys()))
        )

        EV0_branch1_flat_plot = None
        if 1 in self.mu_pipeline.fit_branches:
            EV0_branch1_flat_plot = self.mu_pipeline.ev0_flat(branch_id=1, tau=tau_key_plot)

        EV0_branch2_flat_plot = None
        if 2 in self.mu_pipeline.fit_branches:
            EV0_branch2_flat_plot = self.mu_pipeline.ev0_flat(branch_id=2, tau=tau_key_plot)

        # ----------------------------------------------------------
        # 2) Prepare main eigenvalue plot
        # ----------------------------------------------------------
        fig, ax, mu_for_R, save_path = initialize_plot(
            window,
            R,
            R_value,
            mu_order,
            mu_array,
            EV0_branch1_flat_plot,
            EV0_branch2_flat_plot,
            self.config,
            tau_plot,
            filename,
            correction,
            branch_id=plot_branch_id,
        )

        # ----------------------------------------------------------
        # 3) Optional TXT reference generation
        # ----------------------------------------------------------
        if save_solution:
            sol = compute_reference_solutions_two_branches(
                F_model,
                n_values,
                tau_plot,
                order,
                self.config,
                window,
            )

            b1, b2 = [], []
            for _, roots in sol:
                if roots is None or len(roots) < 2:
                    raise ValueError("TXT generation failed: missing acoustic roots.")
                b1.append(roots[0])
                b2.append(roots[1])

            save_reference_and_noisy_solutions_two_branches(
                solutions_branch1=np.asarray(b1, dtype=np.complex128),
                solutions_branch2=np.asarray(b2, dtype=np.complex128),
                n_values=n_values,
                noise_sigma=nprandomsigma,
                results_dir="./Results/Solutions/",
                prefix="Reference_case",
            )

        # ----------------------------------------------------------
        # 4) Plot eigenvalue trajectories
        # ----------------------------------------------------------
        plot_eigenvalue_trajectories(
            F_model,
            ax,
            self.config,
            tau_plot,
            order,
            mu_for_R,
            mu_order,
            Galerkin,
            window,
            correction,
            F_model.__name__,
            comparison,
        )

        # ----------------------------------------------------------
        # 5) Overlays
        # ----------------------------------------------------------
        overlay_data_if_requested(
            ax,
            self.config,
            tau_plot,
            R_value,
            window,
            show_tax,
            use_txt_solutions,
        )

        # ----------------------------------------------------------
        # 6) Finalize main figure
        # ----------------------------------------------------------
        ax.legend()

        # ----------------------------------------------------------
        # 7) SAVE MAIN FIGURE *BEFORE* ANY OTHER PLOTS
        # ----------------------------------------------------------
        if save_fig:
            fig.savefig(
                save_path,
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.2,
            )
            self.logger.info(f"Main eigenvalue figure saved to: {save_path}")

        # ----------------------------------------------------------
        # 8) αKν (.mat) vs αKμν (fit) complex-plane comparison
        # ----------------------------------------------------------
        if alphaK_ref_eff is not None and alphaK_fit_eff is not None:
            try:
                alphaK_figs = plot_alphaK_comparison_complex_plane_separate(
                    R=R,
                    ref_eff=alphaK_ref_eff,
                    fit_eff=alphaK_fit_eff,
                    title_suffix=f"({self.config.name})",
                    show_labels=True,
                    connect_pairs=False,
                )

                if save_fig:
                    compare_dir = "./Results/Figures"
                    os.makedirs(compare_dir, exist_ok=True)

                    for comp, (fig_comp, _) in alphaK_figs.items():
                        compare_path = os.path.join(
                            compare_dir,
                            f"alphaK_nu_vs_alphaK_mu_nu_{self.config.name}_{comp}_{int(round(tau_plot * 1000))}ms.png",
                        )

                        fig_comp.savefig(
                            compare_path,
                            dpi=600,
                            bbox_inches="tight",
                            pad_inches=0.2,
                        )
                        self.logger.info(
                            f"αKν vs αKμν comparison figure for component {comp} saved to: {compare_path}"
                        )

            except Exception as e:
                self.logger.warning(
                    f"Could not generate αKν vs αKμν comparison figure: {e}"
                )

        # ----------------------------------------------------------
        # 9) Global μ(R) diagnostics (creates its OWN figures)
        # ----------------------------------------------------------
        # plot_mu_global_diagnostics(
        #     mu_array=mu_array,
        #     R=R,
        #     config=self.config,
        #     mu_pipeline=self.mu_pipeline,
        #     show_fig=show_fig,
        #     save_fig=save_fig,
        #     fig_dir="./Results/Figures",
        #     title_suffix="(second-order fit)",
        # )

        # ----------------------------------------------------------
        # 10) SHOW FIGURES
        # ----------------------------------------------------------
        if show_fig:
            import matplotlib.pyplot as plt
            plt.show()