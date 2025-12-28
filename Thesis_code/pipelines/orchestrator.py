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


class SolveEigenWorkflow:
    """
    High-level orchestrator for:
      - μ fitting (via MUFITPipeline provided from outside)
      - solving eigenvalues
      - saving TXT reference/noisy data
      - plotting and overlays
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
        tau: float,
        order: int,
        use_only_acoustic: bool,
        enforce_symmetry: bool,
        save_mu: bool,
        use_saved_mu: bool,
    ):
        """
        Compute or load μ-array and corresponding R, EV0.
        """

        if not correction:
            mu_array = 1.0
            EV0_flat = np.hstack(self.mu_pipeline.EV0).ravel()
            R = self.mu_pipeline.R
            self.logger.info("No correction: using μ = 1.0")
            return mu_array, R, EV0_flat

        if not use_saved_mu:
            if mu_order == "First":
                lin_results = self.mu_pipeline.find_mu_fit_linear_selected_branches(
                    self.config, tau
                )

                if len(lin_results) == 1:
                    branch_id = next(iter(lin_results.keys()))
                    res = lin_results[branch_id]
                    mu_array = res["mu"]
                    R = res["R"]
                    EV0_flat = res["EV0"]

                    self.logger.info(
                        f"Linear μ-fit completed for branch {branch_id} only."
                    )
                else:
                    mu_array = lin_results
                    R = next(iter(lin_results.values()))["R"]
                    EV0_flat = None

                    self.logger.info(
                        f"Linear μ-fit completed for branches {list(lin_results.keys())}."
                    )

            elif mu_order == "Second":
                mu_array, R, EV0_flat = self.mu_pipeline.find_mu_fit_second_order(
                    self.config,
                    tau,
                    order,
                    use_only_acoustic,
                    enforce_symmetry,
                )
            else:
                raise ValueError(f"Invalid mu_order: {mu_order}")

            if save_mu:
                self._save_mu_values(
                    mu_array,
                    R,
                    EV0_flat,
                    flame_model_approximator,
                    tau,
                )
        else:
            mu_array, R, EV0_flat = self._load_mu_values(
                flame_model_approximator,
                tau,
            )

        return mu_array, R, EV0_flat

    # ----------------------------------------------------------
    def _save_mu_values(self, mu_array, R, EV0_flat, flame_model_approximator, tau):
        save_dir = "./Results/Mu_values"
        os.makedirs(save_dir, exist_ok=True)

        base = (
            f"mu_values_{flame_model_approximator}_"
            f"{self.config.name}_{int(tau*1000)}_ms"
        )

        np.savetxt(
            os.path.join(save_dir, base),
            mu_array.astype(str),
            fmt="%s",
            delimiter=",",
        )

        ev0_path = f"./data/{int(tau*1000)}ms/"
        os.makedirs(ev0_path, exist_ok=True)

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

        self.logger.info("Saved μ-array, EV0, and R tensors.")

    # ----------------------------------------------------------
    def _load_mu_values(self, flame_model_approximator, tau):
        base = (
            f"./Results/Mu_values/"
            f"mu_values_{flame_model_approximator}_{self.config.name}_{int(tau*1000)}_ms"
        )

        mu_array = np.loadtxt(base, dtype=complex, delimiter=",")

        ev0_path = f"./data/{int(tau*1000)}ms/"
        EV0_flat = np.loadtxt(
            os.path.join(ev0_path, f"{self.config.name}_EV0_values"),
            dtype=complex,
            delimiter=",",
        )
        R = np.loadtxt(
            os.path.join(ev0_path, f"{self.config.name}_R_values"),
            dtype=float,
            delimiter=",",
        )

        self.logger.info("Loaded μ-array, EV0, and R from disk.")
        return mu_array, R, EV0_flat

    # ----------------------------------------------------------
    def run(
        self,
        correction: bool,
        order: int,
        mu_order: str,
        F_model,
        tau: float,
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
        use_only_acoustic: bool,
        use_txt_solutions: bool,
        enforce_symmetry: bool,
        nprandomsigma: float,
        comparison: bool,
    ):
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
            self.logger.info(
                f"Linear μ plotting will use acoustic branch {plot_branch_id}."
            )

        # ----------------------------------------------------------
        # 1) Compute μ
        # ----------------------------------------------------------
        mu_array, R, EV0_flat = self._compute_mu(
            correction,
            mu_order,
            F_model.__name__,
            tau,
            order,
            use_only_acoustic,
            enforce_symmetry,
            save_mu,
            use_saved_mu,
        )

        # Flatten branch-2 EV0 if available
        EV0_branch2_flat = None
        if getattr(self.mu_pipeline, "EV0_branch2", None) is not None:
            try:
                EV0_branch2_flat = np.hstack(self.mu_pipeline.EV0_branch2).ravel()
            except Exception as exc:
                self.logger.error(f"Failed to flatten EV0_branch2: {exc}")

        # ----------------------------------------------------------
        # 2) Prepare main eigenvalue plot
        # ----------------------------------------------------------
        fig, ax, mu_for_R, save_path = initialize_plot(
            window,
            R,
            R_value,
            mu_order,
            mu_array,
            EV0_flat,
            EV0_branch2_flat,
            self.config,
            tau,
            filename,
            correction,
            enforce_symmetry,
            branch_id=plot_branch_id,
        )

        # ----------------------------------------------------------
        # 3) Optional TXT reference generation
        # ----------------------------------------------------------
        if save_solution:
            sol = compute_reference_solutions_two_branches(
                F_model,
                n_values,
                tau,
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
            tau,
            order,
            mu_for_R,
            mu_order,
            Galerkin,
            window,
            correction,
            enforce_symmetry,
            F_model.__name__,
            comparison,
        )

        # ----------------------------------------------------------
        # 5) Overlays
        # ----------------------------------------------------------
        overlay_data_if_requested(
            ax,
            self.config,
            tau,
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
            # save_path is already a FULL file path → do NOT rebuild it
            fig.savefig(
                save_path,
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.2,
            )

            self.logger.info(f"Main eigenvalue figure saved to: {save_path}")

        # ----------------------------------------------------------
        # 8) Global μ(R) diagnostics (creates its OWN figures)
        # ----------------------------------------------------------
        # plot_mu_global_diagnostics(
        #     mu_array=mu_array,
        #     R=R,
        #     config=self.config,
        #     mu_pipeline=self.mu_pipeline,
        #     enforce_symmetry=enforce_symmetry,
        #     show_fig=show_fig,
        #     save_fig=save_fig,
        #     fig_dir="./Results/Figures",
        #     title_suffix="(second-order fit)",
        # )

        # ----------------------------------------------------------
        # 9) SHOW FIGURES
        # ----------------------------------------------------------
        if show_fig:
            import matplotlib.pyplot as plt
            plt.show()
