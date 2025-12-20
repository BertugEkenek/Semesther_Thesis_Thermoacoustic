# pipelines/simulation.py

import numpy as np
from solver.solver import create_solution_data, create_solution_data_two_branches


def compute_reference_solutions(
    F_model,
    n_values: np.ndarray,
    tau: float,
    order: int,
    config: object,
    Galerkin: str,
    window: int,
):
    """
    Backward compatible: compute single-branch acoustic reference solutions.
    """
    return create_solution_data(
        F_model,
        n_values,
        tau,
        order,
        config,
        correction=False,
        Galerkin=Galerkin,
        window=window,
    )


def compute_reference_solutions_two_branches(
    F_model,
    n_values: np.ndarray,
    tau: float,
    order: int,
    config: object,
    window: int,
):
    """
    NEW: compute two-branch acoustic reference solutions.
    """
    return create_solution_data_two_branches(
        F_model,
        n_values,
        tau,
        order,
        config,
        window=window,
    )
