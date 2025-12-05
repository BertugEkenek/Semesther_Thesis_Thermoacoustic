# pipelines/orchestrator.py

import os
import numpy as np

from utils import logger
from pipelines.prep_work import initialize_plot        # keep only this
from pipelines.simulation import compute_reference_solutions
from pipelines.io_pipeline import save_reference_and_noisy_solutions
from pipelines.plotting_pipeline import (
    plot_eigenvalue_trajectories,
    overlay_data_if_requested,
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

        # ---------------------------------------
        # With correction
        # ---------------------------------------
        if not use_saved_mu:
            if mu_order == "First":
                mu_array, R, EV0_flat = self.mu_pipeline.find_mu_fit(
                    self.config, tau
                )
            elif mu_order == "Second":
                mu_array, R, EV0_flat = (
                    self.mu_pipeline.find_mu_fit_second_order(
                        self.config,
                        tau,
                        order,
                        use_only_acoustic,
                        enforce_symmetry,
                    )
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
        F_model,                      # <-- INJECTED FROM MAIN.PY
        tau: float,
        tolerance: int,
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
    ):

        # 1) Compute μ tensor
        mu_array, R, EV0_flat = self._compute_mu(
            correction,
            mu_order,
            F_model.__name__,  # used only for naming files
            tau,
            order,
            use_only_acoustic,
            enforce_symmetry,
            save_mu,
            use_saved_mu,
        )

        # 2) Prepare main plot (extract μ(R = R_value))
        fig, ax, mu_for_R, save_path = initialize_plot(
            tolerance,
            R,
            R_value,
            mu_order,
            mu_array,
            EV0_flat,
            self.config,
            tau,
            filename,
            correction,
            enforce_symmetry,
        )

        # 3) Optionally compute TXT eigenvalue reference solutions
        if save_solution:
            solutions = compute_reference_solutions(
                F_model,
                n_values,
                tau,
                order,
                self.config,
                Galerkin,
                tolerance,
            )

            solutions_array = np.array(
                [s[1][0] for s in solutions],
                dtype=np.complex128,
            )

            save_reference_and_noisy_solutions(
                solutions_array,
                n_values,
                noise_sigma=nprandomsigma,
                results_dir="./Results/Solutions/",
            )

        # 4) Plot eigenvalue trajectories
        plot_eigenvalue_trajectories(
            F_model,
            ax,
            self.config,
            tau,
            order,
            mu_for_R,
            mu_order,
            Galerkin,
            tolerance,
            correction,
            enforce_symmetry,
            F_model.__name__,
        )

        # 5) Experimental overlays
        overlay_data_if_requested(
            ax,
            self.config,
            tau,
            R_value,
            tolerance,
            show_tax,
            use_txt_solutions,
        )

        # 6) Finalize
        ax.legend()

        if save_fig:
            fig.tight_layout()
            fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)

        if show_fig:
            import matplotlib.pyplot as plt
            plt.show()

        return fig, ax
