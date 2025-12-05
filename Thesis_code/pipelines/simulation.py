# pipelines/simulation.py

import numpy as np

from solver.solver import create_solution_data


def compute_reference_solutions(
    F_model,
    n_values: np.ndarray,
    tau: float,
    order: int,
    config: object,
    Galerkin: str,
    tolerance: int,
):
    """
    Compute reference acoustic eigenvalue trajectories (without correction),
    using the existing create_solution_data function.
    """

    solutions = create_solution_data(
        F_model,
        n_values,
        tau,
        order,
        config,
        correction=False,
        Galerkin=Galerkin,
        tolerance=tolerance,
    )
    return solutions
