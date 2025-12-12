# pipelines/plotting_pipeline.py

from plotting.plotting import plot_roots_over_n
from dataio.experimental import (
    overlay_experimental_eigenvalues,
    overlay_reference_eigenvalues,
)


def plot_eigenvalue_trajectories(
    F_model,
    ax,
    config,
    tau: float,
    order: int,
    mu,
    mu_order: str,
    Galerkin: str,
    tolerance: int,
    correction: bool,
    enforce_symmetry: bool,
    flame_model_approximator: str,
):
    """
    Replicates the branch logic from the original SolveforEig.solve_save_plot,
    but isolated from IO and orchestration.
    """

    if Galerkin == "First" and correction and mu_order == "First":
        plot_roots_over_n(
            F_model,
            tau,
            order,
            config,
            mu,
            mu_order,
            Galerkin,
            tolerance,
            correction,
            enforce_symmetry,
            label=f"{flame_model_approximator} Approx with correction μ",
            cmap="jet",
        )
        # plot_roots_over_n(
        #     F_model,
        #     tau,
        #     order,
        #     config,
        #     mu,
        #     mu_order,
        #     Galerkin,
        #     tolerance,
        #     False,
        #     enforce_symmetry,
        #     label=f"{flame_model_approximator} Approx",
        #     cmap="jet",
        # )

    elif Galerkin == "First" and not correction:
        mu = 1.0
        plot_roots_over_n(
            F_model,
            tau,
            order,
            config,
            mu,
            mu_order,
            Galerkin,
            tolerance,
            correction,
            enforce_symmetry,
            label=f"{flame_model_approximator} Approx",
            cmap="jet",
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
            tolerance,
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
            tolerance,
            correction,
            enforce_symmetry,
            label=f"{flame_model_approximator} Approx with correction μ",
            cmap="jet",
        )

    else:
        import logging

        logging.warning("Invalid input configuration.")


def overlay_data_if_requested(
    ax,
    config,
    tau: float,
    R_value: float,
    tolerance: int,
    show_tax: bool,
    use_txt_solutions: bool,
):
    """
    Overlay experimental and/or reference eigenvalues if requested.
    """

    if show_tax:
        mat_file = f"./data/tax_data/{int(tau*1000)}ms/R={R_value}/tax_{config.name}.mat"
        overlay_experimental_eigenvalues(
            ax,
            mat_file,
            omega_ref=config.w[0].imag,
            tolerance=tolerance / 2,
        )

    if use_txt_solutions and hasattr(config, "txt_solution_path"):
        overlay_reference_eigenvalues(
            ax,
            config.txt_solution_path,
            omega_ref=config.w[0].imag,
            tolerance=tolerance / 2,
        )
