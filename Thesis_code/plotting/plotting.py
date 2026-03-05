# plotting/plotting.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from solver.solver import solve_roots


def plot_roots_over_n(
    F_model,
    tau: float,
    order: int,
    config,
    mu: complex,
    mu_order: str,
    Galerkin: str,
    window: int,
    correction: bool,
    label: str,
    cmap="jet",
    branch_id=None,
    marker='o'
):
    """
    Plot the eigenvalues of a given flame model over a range of n values.
    """
    n_vals = np.linspace(0.001, 4.0, 31)
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
            branch_id=branch_id,
        )

        omega_sigma = [
            (r.imag, r.real)
            for r in roots_set
            if (
                abs(r.imag - w1) < window
                and abs(r.imag - w2) < window
                and r.imag > 0
                and r.imag < 4000
            )
            and abs(r.real) < 550
        ]

        if omega_sigma:
            omega, sigma = zip(*omega_sigma)
            plt.scatter(
                omega,
                sigma,
                color=color,
                marker=marker,
                s=10,
                label=label if n == n_vals[0] else "",
            )

def plot_roots_over_n_new(
    Solutions: list, config: object, label: str, window: int, cmap=plt.cm.copper_r
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
                abs(r.imag - w1) < window / 2
                or abs(r.imag - w2) < window / 2
            )
            and abs(r.real) < window
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

def plot_mu_complex_plane(mu_array, R, title_suffix=""):
    """
    Plot μ values in the complex plane, colored by R.
    Symmetry is inferred from mu_array.shape[1]:
      p_dim=6 => mu21=mu12 (symmetric)
      p_dim=8 => mu21 independent (full)
    """
    R = np.asarray(R)
    if not isinstance(mu_array, np.ndarray) or mu_array.ndim != 2:
        raise ValueError("plot_mu_complex_plane expects mu_array as a 2D ndarray.")

    num_R, p_dim = mu_array.shape

    mu11 = mu_array[:, 0] + 1j * mu_array[:, 1]
    mu22 = mu_array[:, 2] + 1j * mu_array[:, 3]
    mu12 = mu_array[:, 4] + 1j * mu_array[:, 5]

    symmetric = (p_dim == 6)

    mu21 = None
    if not symmetric:
        if p_dim != 8:
            raise ValueError(f"Expected mu_array with p_dim=6 or 8, got {p_dim}.")
        mu21 = mu_array[:, 6] + 1j * mu_array[:, 7]

    cmap = "viridis"
    norm = plt.Normalize(R.min(), R.max())

    fig, axes = plt.subplots(1, 3 if symmetric else 4, figsize=(16, 5), constrained_layout=True)

    def _scatter(ax, mu, label):
        sc = ax.scatter(
            mu.real,
            mu.imag,
            c=R,
            cmap=cmap,
            norm=norm,
            s=40,
            edgecolors="k",
            linewidths=0.3,
        )
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        return sc

    sc = _scatter(axes[0], mu11, r"$\mu_{11}$")
    _scatter(axes[1], mu22, r"$\mu_{22}$")
    _scatter(axes[2], mu12, r"$\mu_{12}$")
    if not symmetric:
        _scatter(axes[3], mu21, r"$\mu_{21}$")

    cbar = fig.colorbar(sc, ax=axes, shrink=0.85)
    cbar.set_label("R")

    fig.suptitle(f"μ values in complex plane {title_suffix}", fontsize=14)
    plt.show()
# ==========================================================
# μ(R) plots (raw parameters)
# ==========================================================
def plot_mu_components(R, mu_array, ax=None):
    """
    Plot μ components vs R.

    p_dim=6 (symmetric):
      [Re11, Im11, Re22, Im22, Re12, Im12]
    p_dim=8 (full):
      [Re11, Im11, Re22, Im22, Re12, Im12, Re21, Im21]
    """
    if ax is None:
        _, ax = plt.subplots()

    R = np.asarray(R)
    mu_array = np.asarray(mu_array)
    if mu_array.ndim != 2:
        raise ValueError("plot_mu_components expects mu_array as 2D array.")

    p_dim = mu_array.shape[1]
    if p_dim not in (6, 8):
        raise ValueError(f"Expected p_dim 6 or 8, got {p_dim}.")

    ax.plot(R, mu_array[:, 0], label="Re(μ11)")
    ax.plot(R, mu_array[:, 1], label="Im(μ11)")
    ax.plot(R, mu_array[:, 2], label="Re(μ22)")
    ax.plot(R, mu_array[:, 3], label="Im(μ22)")
    ax.plot(R, mu_array[:, 4], "--", label="Re(μ12)")
    ax.plot(R, mu_array[:, 5], "--", label="Im(μ12)")

    if p_dim == 8:
        ax.plot(R, mu_array[:, 6], "--", label="Re(μ21)")
        ax.plot(R, mu_array[:, 7], "--", label="Im(μ21)")

    ax.set_xlabel("R")
    ax.set_ylabel("μ")
    ax.set_title("μ(R)")
    ax.legend()
    ax.grid(True)
    return ax
# ==========================================================
# μ·ν(R) plots (effective coupling)
# ==========================================================
def plot_mu_nu_components(R, mu_array, nu, ax=None):
    """
    Plot effective coupling μ_ij * ν_ij vs R (real & imaginary parts).

    mu_array p_dim=6 or 8.
    nu = [ν11, ν22, ν12, ν21]
    """
    if ax is None:
        _, ax = plt.subplots()

    R = np.asarray(R)
    mu_array = np.asarray(mu_array)
    nu = np.asarray(nu, dtype=float)

    p_dim = mu_array.shape[1]
    if p_dim not in (6, 8):
        raise ValueError(f"Expected p_dim 6 or 8, got {p_dim}.")

    mu11 = mu_array[:, 0] + 1j * mu_array[:, 1]
    mu22 = mu_array[:, 2] + 1j * mu_array[:, 3]
    mu12 = mu_array[:, 4] + 1j * mu_array[:, 5]

    eff11 = mu11 * nu[0]
    eff22 = mu22 * nu[1]
    eff12 = mu12 * nu[2]

    ax.plot(R, eff11.real, label="Re(μ11 ν11)")
    ax.plot(R, eff11.imag, label="Im(μ11 ν11)")
    ax.plot(R, eff22.real, label="Re(μ22 ν22)")
    ax.plot(R, eff22.imag, label="Im(μ22 ν22)")
    ax.plot(R, eff12.real, "--", label="Re(μ12 ν12)")
    ax.plot(R, eff12.imag, "--", label="Im(μ12 ν12)")

    if p_dim == 8:
        mu21 = mu_array[:, 6] + 1j * mu_array[:, 7]
        eff21 = mu21 * nu[3]
        ax.plot(R, eff21.real, "--", label="Re(μ21 ν21)")
        ax.plot(R, eff21.imag, "--", label="Im(μ21 ν21)")

    ax.set_xlabel("R")
    ax.set_ylabel("Effective coupling μ·ν")
    ax.set_title("Effective thermoacoustic coupling")
    ax.legend()
    ax.grid(True)
    return ax


def plot_mu_nu_magnitude(R, mu_array, nu, ax=None):
    """
    Plot |μ_ij ν_ij| vs R. Supports p_dim=6 or 8.
    """
    if ax is None:
        _, ax = plt.subplots()

    R = np.asarray(R)
    mu_array = np.asarray(mu_array)
    nu = np.asarray(nu, dtype=float)

    p_dim = mu_array.shape[1]
    if p_dim not in (6, 8):
        raise ValueError(f"Expected p_dim 6 or 8, got {p_dim}.")

    mu11 = mu_array[:, 0] + 1j * mu_array[:, 1]
    mu22 = mu_array[:, 2] + 1j * mu_array[:, 3]
    mu12 = mu_array[:, 4] + 1j * mu_array[:, 5]

    ax.plot(R, np.abs(mu11 * nu[0]), label="|μ11 ν11|")
    ax.plot(R, np.abs(mu22 * nu[1]), label="|μ22 ν22|")
    ax.plot(R, np.abs(mu12 * nu[2]), label="|μ12 ν12|")

    if p_dim == 8:
        mu21 = mu_array[:, 6] + 1j * mu_array[:, 7]
        ax.plot(R, np.abs(mu21 * nu[3]), label="|μ21 ν21|")

    ax.set_xlabel("R")
    ax.set_ylabel("|μ·ν|")
    ax.set_title("Magnitude of effective coupling")
    ax.legend()
    ax.grid(True)
    return ax

# ==========================================================
# Conditioning diagnostics
# ==========================================================
def plot_condition_numbers(R, cond_numbers, ax=None):
    """
    Plot condition number of A_phys vs R (log scale).
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.semilogy(R, cond_numbers, marker="o")
    ax.set_xlabel("R")
    ax.set_ylabel("cond(A)")
    ax.set_title("Condition number of second-order system")
    ax.grid(True, which="both")
    return ax


def plot_ranks(R, ranks, ax=None):
    """
    Plot rank of A_phys vs R.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(R, ranks, marker="s")
    ax.set_xlabel("R")
    ax.set_ylabel("Rank")
    ax.set_title("Rank of second-order system")
    ax.grid(True)
    return ax


# ==========================================================
# Residual diagnostics
# ==========================================================
def plot_residuals(R, residuals, ax=None):
    """
    Plot residual norm vs R.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(R, residuals, marker="x")
    ax.set_xlabel("R")
    ax.set_ylabel("Residual norm")
    ax.set_title("Second-order fit residuals")
    ax.grid(True)
    return ax

def plot_mu_nu_complex_plane(mu_array, R, nu, title_suffix=""):
    """
    Plot effective coupling μ·ν values in the complex plane, colored by R.

    mu_array p_dim=6 => symmetric (no μ21 plot)
    mu_array p_dim=8 => includes μ21 plot
    """
    R = np.asarray(R)
    nu = np.asarray(nu, dtype=float)

    if not isinstance(mu_array, np.ndarray) or mu_array.ndim != 2:
        raise ValueError("plot_mu_nu_complex_plane expects mu_array as a 2D ndarray.")

    num_R, p_dim = mu_array.shape
    if num_R != len(R):
        raise ValueError(f"R length mismatch: len(R)={len(R)} vs mu_array rows={num_R}.")

    if p_dim not in (6, 8):
        raise ValueError(f"Expected p_dim 6 or 8, got {p_dim}.")

    mu11 = mu_array[:, 0] + 1j * mu_array[:, 1]
    mu22 = mu_array[:, 2] + 1j * mu_array[:, 3]
    mu12 = mu_array[:, 4] + 1j * mu_array[:, 5]

    eff11 = mu11 * nu[0]
    eff22 = mu22 * nu[1]
    eff12 = mu12 * nu[2]

    eff21 = None
    if p_dim == 8:
        mu21 = mu_array[:, 6] + 1j * mu_array[:, 7]
        eff21 = mu21 * nu[3]

    cmap = "viridis"
    norm = plt.Normalize(R.min(), R.max())

    ncols = 3 if p_dim == 6 else 4
    fig, axes = plt.subplots(1, ncols, figsize=(16, 5), constrained_layout=True)

    def _scatter(ax, z, label):
        sc = ax.scatter(
            z.real,
            z.imag,
            c=R,
            cmap=cmap,
            norm=norm,
            s=40,
            edgecolors="k",
            linewidths=0.3,
        )
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        return sc

    sc = _scatter(axes[0], eff11, r"$\mu_{11}\nu_{11}$")
    _scatter(axes[1], eff22, r"$\mu_{22}\nu_{22}$")
    _scatter(axes[2], eff12, r"$\mu_{12}\nu_{12}$")

    if p_dim == 8:
        _scatter(axes[3], eff21, r"$\mu_{21}\nu_{21}$")

    cbar = fig.colorbar(sc, ax=axes, shrink=0.85)
    cbar.set_label("R")

    fig.suptitle(f"Effective coupling (μ·ν) in complex plane {title_suffix}", fontsize=14)
    return fig, axes