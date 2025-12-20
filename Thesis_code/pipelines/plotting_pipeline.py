# pipelines/plotting_pipeline.py

import os
import numpy as np
import matplotlib.pyplot as plt

from utils import logger

from plotting.plotting import (
    plot_roots_over_n,
    plot_mu_complex_plane,
    plot_mu_nu_components,
    plot_mu_nu_magnitude,
    plot_condition_numbers,
    plot_ranks,
    plot_residuals,
    plot_mu_nu_complex_plane,
)

from dataio.experimental import (
    overlay_experimental_eigenvalues,
    overlay_reference_eigenvalues,
)


# ==========================================================
# Eigenvalue trajectory plotting
# ==========================================================
def plot_eigenvalue_trajectories(
    F_model,
    ax,
    config,
    tau: float,
    order: int,
    mu,
    mu_order: str,
    Galerkin: str,
    window: int,
    correction: bool,
    enforce_symmetry: bool,
    flame_model_approximator: str,
    comparison: bool,
):
    """
    Plot eigenvalue trajectories for the selected configuration.

    Replicates the branch logic from the original SolveforEig.solve_save_plot,
    but isolated from IO and orchestration.
    """

    # ----------------------------------------------------------
    # Determine branch_id for characteristic evaluation
    # ----------------------------------------------------------
    branch_id = getattr(ax, "_plot_branch_id", None)

    if mu_order == "First":
        if branch_id not in (1, 2):
            raise ValueError(
                "plot_eigenvalue_trajectories: branch_id must be set on axis "
                "for linear μ plotting."
            )

        logger.info(
            f"[plot_eigenvalue_trajectories] Using branch {branch_id} "
            "for linear characteristic evaluation."
        )
    else:
        # second-order μ → branch handled internally
        branch_id = None

    # ----------------------------------------------------------
    # Galerkin / correction logic
    # ----------------------------------------------------------
    if Galerkin == "First" and correction and mu_order == "First":
        plot_roots_over_n(
            F_model,
            tau,
            order,
            config,
            mu,
            mu_order,
            Galerkin,
            window,
            correction,
            enforce_symmetry,
            label=f"{flame_model_approximator} Approx with correction μ",
            cmap="jet",
            branch_id=branch_id,
        )

        if comparison:
            plot_roots_over_n(
                F_model,
                tau,
                order,
                config,
                mu=1.0,
                mu_order="First",
                Galerkin="First",
                window=window,
                correction=False,
                enforce_symmetry=False,
                label=f"{flame_model_approximator} Approx",
                cmap="copper",
                branch_id=branch_id,
                marker="+",
            )

    elif Galerkin == "First" and not correction:
        plot_roots_over_n(
            F_model,
            tau,
            order,
            config,
            mu=1.0,
            mu_order=mu_order,
            Galerkin=Galerkin,
            window=window,
            correction=False,
            enforce_symmetry=False,
            label=f"{flame_model_approximator} Approx",
            cmap="jet",
            branch_id=branch_id,
        )

    elif Galerkin == "Second" and not correction:
        plot_roots_over_n(
            F_model,
            tau,
            order,
            config,
            mu,
            mu_order,
            Galerkin,
            window,
            correction,
            enforce_symmetry,
            label=f"{flame_model_approximator} Approx",
            cmap="jet",
        )

    elif Galerkin == "Second" and correction and mu_order == "Second":
        plot_roots_over_n(
            F_model,
            tau,
            order,
            config,
            mu,
            mu_order,
            Galerkin,
            window,
            correction,
            enforce_symmetry,
            label=f"{flame_model_approximator} Approx with correction μ",
            cmap="jet",
        )

        if comparison:
            plot_roots_over_n(
                F_model,
                tau,
                order,
                config,
                mu=1.0,
                mu_order=mu_order,
                Galerkin=Galerkin,
                window=window,
                correction=False,
                enforce_symmetry=False,
                label=f"{flame_model_approximator} Approx",
                cmap="copper",
                marker="+",
            )

    else:
        logger.warning("plot_eigenvalue_trajectories: invalid input configuration.")


# ==========================================================
# Experimental / reference overlays
# ==========================================================
def overlay_data_if_requested(
    ax,
    config,
    tau: float,
    R_value: float,
    window: int,
    show_tax: bool,
    use_txt_solutions: bool,
):
    """
    Overlay experimental and/or reference eigenvalues if requested.
    """

    if show_tax:
        mat_file = (
            f"./data/tax_data/{int(tau*1000)}ms/R={R_value}/"
            f"tax_{config.name}_first_branch_up_to_n=4_with_number_of_n=31_"
            f"tau={int(tau*1000)}ms.mat"
        )

        overlay_experimental_eigenvalues(
            ax,
            mat_file,
            omega_ref=config.w[0].imag,
            window=window / 2,
        )

    if use_txt_solutions and hasattr(config, "txt_solution_path"):
        overlay_reference_eigenvalues(
            ax,
            "./Results/Solutions/Reference_case_branch1.txt",
            omega_ref=config.w[0].imag,
            window=window / 2,
        )
        overlay_reference_eigenvalues(
            ax,
            "./Results/Solutions/Reference_case_branch2.txt",
            omega_ref=config.w[1].imag,
            window=window / 2,
        )


# ==========================================================
# Global μ(R) diagnostics
# ==========================================================
def plot_mu_global_diagnostics(
    mu_array,
    R,
    config,
    mu_pipeline=None,
    enforce_symmetry=True,
    show_fig=True,
    save_fig=False,
    fig_dir="./Results/Figures",
    title_suffix="",
):
    """
    Global μ(R) diagnostics and result plots.

    Non-intrusive:
      - no solver calls
      - no pipeline mutation
      - respects show_fig / save_fig flags
    """

    # Guard: only second-order μ arrays
    if not isinstance(mu_array, np.ndarray) or mu_array.ndim != 2:
        return

    os.makedirs(fig_dir, exist_ok=True)

    # --------------------------------------------------
    # μ complex plane (legacy diagnostic; shows immediately)
    # --------------------------------------------------
    plot_mu_complex_plane(
        mu_array=mu_array,
        R=R,
        enforce_symmetry=enforce_symmetry,
        title_suffix=title_suffix,
    )

    # --------------------------------------------------
    # μ·ν components (Re/Im)
    # --------------------------------------------------
    fig1, ax1 = plt.subplots()
    plot_mu_nu_components(R, mu_array, config.nu, ax=ax1)
    fig1.tight_layout()
    if save_fig:
        fig1.savefig(os.path.join(fig_dir, "mu_nu_components.png"), dpi=300)

    # --------------------------------------------------
    # μ·ν magnitude
    # --------------------------------------------------
    fig2, ax2 = plt.subplots()
    plot_mu_nu_magnitude(R, mu_array, config.nu, ax=ax2)
    fig2.tight_layout()
    if save_fig:
        fig2.savefig(os.path.join(fig_dir, "mu_nu_magnitude.png"), dpi=300)
    
    # --------------------------------------------------
    # μ·ν complex plane (effective coupling)
    # --------------------------------------------------
    fig0, _ = plot_mu_nu_complex_plane(
        mu_array=mu_array,
        R=R,
        nu=config.nu,
        enforce_symmetry=enforce_symmetry,
        title_suffix=title_suffix,
    )
    if save_fig:
        fig0.savefig(os.path.join(fig_dir, "mu_nu_complex_plane.png"), dpi=300, bbox_inches="tight")


    # --------------------------------------------------
    # Conditioning diagnostics (optional)
    # --------------------------------------------------
    if mu_pipeline is not None:
        cond_nums = getattr(mu_pipeline, "mu_cond_numbers", None)
        if cond_nums is not None and len(cond_nums) == len(R):
            fig3, ax3 = plt.subplots()
            plot_condition_numbers(R, cond_nums, ax=ax3)
            fig3.tight_layout()
            if save_fig:
                fig3.savefig(os.path.join(fig_dir, "mu_condition_numbers.png"), dpi=300)


        ranks = getattr(mu_pipeline, "mu_ranks", None)
        if ranks is not None and len(ranks) == len(R):
            fig4, ax4 = plt.subplots()
            plot_ranks(R, ranks, ax=ax4)
            fig4.tight_layout()
            if save_fig:
                fig4.savefig(os.path.join(fig_dir, "mu_ranks.png"), dpi=300)


        residuals = getattr(mu_pipeline, "mu_residuals", None)
        if residuals is not None and len(residuals) == len(R):
            fig5, ax5 = plt.subplots()
            plot_residuals(R, residuals, ax=ax5)
            fig5.tight_layout()
            if save_fig:
                fig5.savefig(os.path.join(fig_dir, "mu_residuals.png"), dpi=300)

