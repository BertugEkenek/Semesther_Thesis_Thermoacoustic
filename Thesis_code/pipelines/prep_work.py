# pipelines/prep_work.py

import os
from utils import logger
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Plot initialization
# --------------------------------------------------------------
# --------------------------------------------------------------
# Plot initialization
# --------------------------------------------------------------
def initialize_plot(
    tolerance: int,
    R: np.ndarray,
    R_value: float,
    mu_order: str,
    mu_array: np.ndarray,
    EV0_branch1: np.ndarray,
    EV0_branch2: np.ndarray | None,
    config: object,
    tau: float,
    filename: str,
    correction: bool,
    enforce_symmetry: bool,
):
    """
    Initialize plotting and configure base acoustic modes for the solver.

    Behavior:
      - Mode 1 (s_1) always comes from branch-1 EV0 at the selected R.
      - Mode 2 (s_2) comes from branch-2 EV0 if available; otherwise fallback
        to config.w_R_table[R] and finally to config.w[1].

      - μ is taken from mu_array at the closest R index.
      - Plot styling (axis labels, title, grid, save path) preserved.
    """

    logger.info(
        f"Initializing plot for R={R_value}, mu_order={mu_order}, "
        f"correction={correction}"
    )

    # float-safe index for the requested R_value
    index = int(np.argmin(np.abs(R - R_value)))

    fig, ax = plt.subplots(figsize=(10, 6))

    # -------------------------------
    # Set base acoustic modes (s1, s2)
    # -------------------------------
    s_1 = EV0_branch1[index]
    config.w[0] = s_1

    # Mode 2: use branch-2 EV0 if provided, else fallback to table / previous
    if EV0_branch2 is not None:
        s_2 = EV0_branch2[index]
    else:
        try:
            s_2 = config.w_R_table[float(R_value)]
        except Exception:
            s_2 = config.w[1]
            logger.debug(
                f"[initialize_plot] Using config.w[1] as s_2 fallback at R={R_value}."
            )

    config.w[1] = s_2

    logger.info(f"[initialize_plot] s_1 = {s_1}, s_2 = {s_2}")

    # -------------------------------
    # Correction / μ usage
    # -------------------------------
    if correction:
        ax.axhline(0, color="gray", linestyle="--")

        if mu_order == "First":
            mu = mu_array[index][0] + 1j * mu_array[index][1]
            logger.info(f"[initialize_plot] Using first-order μ = {mu}")

            ax.axvline(
                config.w[0].imag,
                color="black",
                linestyle="-",
                linewidth=1,
                label="ω₁",
            )

        elif mu_order == "Second":
            mu = mu_array[index]

            if enforce_symmetry:
                logger.info(
                    "Using symmetric second-order μ values:\n"
                    f"  μ11 = {mu[0]:.4f} + i {mu[1]:.4f}\n"
                    f"  μ22 = {mu[2]:.4f} + i {mu[3]:.4f}\n"
                    f"  μ12 = μ21 = {mu[4]:.4f} + i {mu[5]:.4f}"
                )
            else:
                logger.info(
                    "Using second-order μ values:\n"
                    f"  μ11 = {mu[0]:.4f} + i {mu[1]:.4f}\n"
                    f"  μ22 = {mu[2]:.4f} + i {mu[3]:.4f}\n"
                    f"  μ12 = {mu[4]:.4f} + i {mu[5]:.4f}\n"
                    f"  μ21 = {mu[6]:.4f} + i {mu[7]:.4f}"
                )

            ax.axvline(
                config.w[0].imag,
                color="black",
                linestyle="-",
                linewidth=1,
                label="ω₁",
            )

        else:
            raise ValueError(f"Invalid mu_order '{mu_order}'")

    else:
        mu = 1.0
        logger.info("[initialize_plot] No correction applied, using μ = 1.0")
        ax.axhline(0, color="gray", linestyle="--")
        ax.axvline(
            config.w[0].imag,
            color="black",
            linestyle="-",
            linewidth=1,
            label="ω₁",
        )

    # Labels and title (preserved)
    ax.set_xlabel("Frequency (rad/s)")
    ax.set_ylabel("Growth rate (rad/s)")
    ax.set_title(
        f"Eigenvalues near ω₁ with n ∈ [0.001,4], τ={tau:.5f}s, R={R_value}"
    )
    ax.grid(True)
    plt.tight_layout()

    # Save location (preserved)
    save_dir = f"./Results/Plots/{config.name}/{int(tau * 1000)}ms/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    return fig, ax, mu, save_path


# --------------------------------------------------------------
# EV trajectory conversion
# --------------------------------------------------------------
def prepare_mu(EV_trajectories: np.ndarray, min_size: int):
    """
    Convert EV_trajectories [R, τ, n, trajectories] from object-type entries
    to a numeric complex array safely.
    """

    shape = EV_trajectories.shape
    result = np.zeros(shape, dtype=np.complex128)

    it = np.nditer(result, flags=["multi_index"], op_flags=["writeonly"])

    for x in it:
        idx = it.multi_index
        val = EV_trajectories[idx]

        # Case 1: scalar
        if np.isscalar(val):
            x[...] = complex(val)
            continue

        arr = np.asarray(val)

        # Case 2: 0D numpy scalar
        if arr.ndim == 0:
            x[...] = complex(arr)
            continue

        # Case 3: array → take first eigenvalue
        x[...] = complex(arr.flatten()[0])

    return result
