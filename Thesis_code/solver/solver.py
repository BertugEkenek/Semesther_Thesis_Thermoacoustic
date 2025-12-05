# solver/solver.py

import numpy as np
from models.model_selector import select_characteristic_model


def solve_roots(
    F_model,
    n: float,
    tau: float,
    order: int,
    config: object,
    mu: complex,
    correction: bool,
    Galerkin: str,
    mu_order: str,
    enforce_symmetry: bool,
):
    """
    Compute polynomial coefficients and return its roots.
    Clean version using model-selection registry.
    """

    s = np.poly1d([1, 0])

    # Select the correct characteristic model function
    characteristic_fn = select_characteristic_model(
        correction=correction,
        mu_order=mu_order,
        Galerkin=Galerkin,
    )

    # Call characteristic model
    # (Function signature supports both 1-mode and 2-mode models)
    if characteristic_fn.__name__ == "characteristic_poly_model3_2mode":
        char_eq = characteristic_fn(
            s, F_model, n, tau, order, config, mu, enforce_symmetry
        )
    elif characteristic_fn.__name__ == "characteristic_poly_model3_1mode":
        char_eq = characteristic_fn(
            s, F_model, n, tau, order, config, mu
        ) if correction else characteristic_fn(
            s, F_model, n, tau, order, config
        )
    else:
        char_eq = characteristic_fn(s, F_model, n, tau, order, config)

    coeffs = np.asarray(char_eq.coeffs, dtype=complex)

    if np.max(np.abs(coeffs)) == 0:
        return []

    coeffs = coeffs / np.max(np.abs(coeffs))
    return np.roots(coeffs)


def create_solution_data(
    F_model,
    n_values: np.ndarray,
    tau: float,
    order: int,
    config: object,
    correction: bool,
    Galerkin: str,
    tolerance: int,
):
    """
    Generate clean acoustic eigenvalue trajectories for TXT saving.
    Always returns exactly ONE eigenvalue per n (the acoustic root).
    """

    solutions = []
    w1 = config.w[0].imag
    w2 = config.w[1].imag

    for n in n_values:
        # For reference TXT generation we use the base model (no correction)
        roots = solve_roots(
            F_model,
            n,
            tau,
            order,
            config,
            mu=1.0,
            correction=False,
            Galerkin="Second",
            mu_order="First",
            enforce_symmetry=False,
        )

        if len(roots) == 0:
            solutions.append((n, []))
            continue

        # Filter only roots near physical acoustic branches
        acoustic_candidates = []
        for r in roots:
            if (abs(r.imag - w1) < tolerance / 2) or (abs(r.imag - w2) < tolerance / 2):
                acoustic_candidates.append(r)

        if len(acoustic_candidates) == 0:
            solutions.append((n, []))
            continue

        # Select the root closest to mode 1
        acoustic_root = min(
            acoustic_candidates,
            key=lambda r: abs(r.imag - w1),
        )

        solutions.append((n, [acoustic_root]))

    return solutions
