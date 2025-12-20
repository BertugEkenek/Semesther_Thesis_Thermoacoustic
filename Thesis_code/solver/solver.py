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
    branch_id: int | None = None,
):

    """
    Compute polynomial coefficients and return its roots.
    Clean version using model-selection registry.
    """

    s = np.poly1d([1, 0])

    characteristic_fn = select_characteristic_model(
    correction=correction,
    mu_order=mu_order,
    Galerkin=Galerkin,
    )

    if characteristic_fn.__name__ == "characteristic_poly_model3_2mode":
        # Second-order: branch handled internally
        char_eq = characteristic_fn(
            s, F_model, n, tau, order, config, mu, enforce_symmetry
        )

    elif characteristic_fn.__name__ == "characteristic_poly_model3_1mode":
        if branch_id not in (1, 2):
            raise ValueError(
                "solve_roots: branch_id must be provided for 1-mode characteristic."
            )

        char_eq = characteristic_fn(
            s,
            F_model,
            n,
            tau,
            order,
            config,
            mu,
            branch_id=branch_id,
        )

    else:
        char_eq = characteristic_fn(s, F_model, n, tau, order, config)


    coeffs = np.asarray(char_eq.coeffs, dtype=complex)

    if np.max(np.abs(coeffs)) == 0:
        return []

    coeffs = coeffs / np.max(np.abs(coeffs))
    return np.roots(coeffs)


def _pick_acoustic_root(roots, w_target, window):
    """
    Pick the root whose imag part is closest to w_target among candidates
    within window window. Returns None if no candidate.
    """
    if roots is None or len(roots) == 0:
        return None

    candidates = []
    half = float(window) / 2.0
    for r in roots:
        if abs(r.imag - w_target) < half:
            candidates.append(r)

    if not candidates:
        return None

    return min(candidates, key=lambda r: abs(r.imag - w_target))


def create_solution_data(
    F_model,
    n_values: np.ndarray,
    tau: float,
    order: int,
    config: object,
    correction: bool,
    Galerkin: str,
    window: int,
):
    """
    Backward compatible: returns exactly ONE eigenvalue per n (branch-1 acoustic root).
    """
    solutions = []
    w1 = config.w[0].imag

    for n in n_values:
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
        branch_id=1,
    )


        r1 = _pick_acoustic_root(roots, w1, window)
        if r1 is None:
            solutions.append((n, []))
        else:
            solutions.append((n, [r1]))

    return solutions


def create_solution_data_two_branches(
    F_model,
    n_values: np.ndarray,
    tau: float,
    order: int,
    config: object,
    window: int,
):
    """
    NEW: returns TWO eigenvalues per n (acoustic branch-1 and branch-2),
    selected as roots closest to w1 and w2 respectively.

    Output format:
      solutions: list of (n, [root_branch1, root_branch2]) where missing roots are None.
    """
    solutions = []
    w1 = float(config.w[0].imag)
    w2 = float(config.w[1].imag)

    for n in n_values:
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

        # --- Branch 1 ---
        roots_1 = solve_roots(
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
            branch_id=1,
        )

        # --- Branch 2 ---
        roots_2 = solve_roots(
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
            branch_id=2,
        )

        r1 = _pick_acoustic_root(roots_1, w1, window)
        r2 = _pick_acoustic_root(roots_2, w2, window)

        solutions.append((n, [r1, r2]))

    return solutions
