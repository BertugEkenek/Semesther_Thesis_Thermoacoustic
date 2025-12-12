# plotting/plotting.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from solver.solver import solve_roots  # updated import


def plot_roots_over_n(
    F_model,
    tau: float,
    order: int,
    config,
    mu: complex,
    mu_order: str,
    Galerkin: str,
    tolerance: int,
    correction: bool,
    enforce_symmetry: bool,
    label: str,
    cmap="jet",
):
    """
    Plot the eigenvalues of a given flame model over a range of n values.
    """

    n_vals = np.linspace(0.001, 1.0, 31)
    norm = Normalize(vmin=n_vals[0], vmax=n_vals[-1])
    smap = ScalarMappable(norm=norm, cmap=cmap)

    w1 = config.w[0].imag
    w2 = config.w[1].imag  # not currently used in filter

    for n in n_vals:
        color = smap.to_rgba(n)
        roots_set = solve_roots(
            F_model,
            n,
            tau,
            order,
            config,
            mu,
            correction,
            Galerkin,
            mu_order,
            enforce_symmetry,
        )
        omega_sigma = [
            (r.imag, r.real)
            for r in roots_set
            if (
                abs(r.imag - w1) < tolerance
                and r.imag > 0
                and r.imag < 3000  # hard cutoff for better visualization
            )
            and abs(r.real) < tolerance
        ]
        if omega_sigma:
            omega, sigma = zip(*omega_sigma)
            plt.scatter(
                omega,
                sigma,
                color=color,
                s=10,
                label=label if n == n_vals[0] else "",
            )


def plot_roots_over_n_new(
    Solutions: list, config: object, label: str, tolerance: int, cmap=plt.cm.copper_r
):
    """
    Plot the eigenvalues of a given flame model over a range of n values,
    using a precomputed list of solutions (n, roots_set).
    """

    n_vals = [sol[0] for sol in Solutions]
    norm = Normalize(vmin=n_vals[0], vmax=n_vals[-1])
    smap = ScalarMappable(norm=norm, cmap=cmap)

    w1 = config.w[0].imag
    w2 = config.w[1].imag

    for sol in Solutions:
        n = sol[0]
        roots_set = sol[1]
        color = smap.to_rgba(n)
        omega_sigma = [
            (r.imag, r.real)
            for r in roots_set
            if (
                abs(r.imag - w1) < tolerance / 2
                or abs(r.imag - w2) < tolerance / 2
            )
            and abs(r.real) < tolerance
        ]
        if omega_sigma:
            omega, sigma = zip(*omega_sigma)
            plt.scatter(
                omega,
                sigma,
                color=color,
                s=10,
                label=label if n == n_vals[0] else "",
            )
