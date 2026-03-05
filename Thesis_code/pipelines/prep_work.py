# pipelines/prep_work.py

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import logger

# --------------------------------------------------------------
# Plot initialization
# --------------------------------------------------------------
def initialize_plot(
    window: int,
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
    branch_id: int | None = None,
):
    """
    Initialize plotting and configure base acoustic modes.

    Linear case:
      - branch_id MUST be provided (1 or 2)

    Second-order case:
      - branch_id is ignored
      - symmetry is inferred from μ length:
          len=6 => μ12=μ21
          len=8 => full μ12, μ21
    """

    logger.info(
        f"Initializing plot for R={R_value}, mu_order={mu_order}, "
        f"correction={correction}, branch_id={branch_id}"
    )

    # -------------------------------
    # Locate R index
    # -------------------------------
    index = int(np.argmin(np.abs(R - R_value)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax._plot_branch_id = branch_id

    # ==========================================================
    # Base acoustic modes
    # ==========================================================
    if mu_order == "First":
        if branch_id not in (1, 2):
            raise ValueError(
                "initialize_plot: branch_id must be 1 or 2 for linear μ plotting."
            )

        if branch_id == 1:
            s_ref = EV0_branch1[index]
        else:
            if EV0_branch2 is None:
                raise ValueError("initialize_plot: EV0_branch2 is required for branch_id=2.")
            s_ref = EV0_branch2[index]

        config.w[branch_id - 1] = s_ref
        logger.info(
            f"[initialize_plot | linear] Using branch {branch_id} reference mode s = {s_ref}"
        )

    else:
        # -------------------------------
        # Second order → two branches
        # -------------------------------
        if EV0_branch2 is None:
            raise ValueError("initialize_plot: EV0_branch2 must be provided for second-order plotting.")

        s1 = EV0_branch1[index]
        s2 = EV0_branch2[index]
        config.w[0] = s1
        config.w[1] = s2

        logger.info(f"[initialize_plot | second-order] s1 = {s1}, s2 = {s2}")

    # ==========================================================
    # μ selection and reporting
    # ==========================================================
    if correction:
        ax.axhline(0, color="gray", linestyle="--")

        if mu_order == "First":
            mu = mu_array[index][0] + 1j * mu_array[index][1]
            logger.info(f"[initialize_plot] Using first-order μ = {mu}")

            ax.axvline(
                s_ref.imag,
                color="black",
                linestyle="-",
                linewidth=1,
                label=f"ω (branch {branch_id})",
            )

        elif mu_order == "Second":
            mu = np.asarray(mu_array[index], dtype=float)
            p_dim = int(mu.size)

            if p_dim not in (6, 8):
                raise ValueError(
                    f"initialize_plot: second-order mu must have length 6 or 8, got {p_dim}."
                )

            mur11, mui11 = mu[0], mu[1]
            mur22, mui22 = mu[2], mu[3]
            mur12, mui12 = mu[4], mu[5]

            text = (
                "Using second-order μ values:\n"
                f"  μ11 = {mur11:.4f} + i {mui11:.4f}\n"
                f"  μ22 = {mur22:.4f} + i {mui22:.4f}\n"
                f"  μ12 = {mur12:.4f} + i {mui12:.4f}\n"
            )

            if p_dim == 6:
                text += "  μ21 = μ12 (symmetric)"
            else:
                mur21, mui21 = mu[6], mu[7]
                text += f"  μ21 = {mur21:.4f} + i {mui21:.4f}"

            logger.info(text)

            ax.axvline(
                config.w[0].imag,
                color="black",
                linestyle="-",
                linewidth=1,
                label="ω₁",
            )
            ax.axvline(
                config.w[1].imag,
                color="black",
                linestyle="-",
                linewidth=1,
                label="ω₂",
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
            label="ω",
        )

    # ==========================================================
    # Labels, title, save path
    # ==========================================================
    ax.set_xlabel("Frequency (rad/s)")
    ax.set_ylabel("Growth rate (rad/s)")
    ax.set_title(
        f"Eigenvalues near ω with n ∈ [0.001,4], τ={tau:.5f}s, R={R_value}"
    )
    ax.grid(True)
    plt.tight_layout()

    save_dir = os.path.join(".", "Results", "Plots", config.name, f"{int(tau * 1000)}ms")
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(filename)
    save_path = os.path.join(save_dir, filename)

    return fig, ax, mu, save_path
# --------------------------------------------------------------
# EV trajectory conversion (unchanged)
# --------------------------------------------------------------
def prepare_mu(EV_trajectories: np.ndarray, min_size: int):
    shape = EV_trajectories.shape
    result = np.zeros(shape, dtype=np.complex128)

    it = np.nditer(result, flags=["multi_index"], op_flags=["writeonly"])

    for x in it:
        idx = it.multi_index
        val = EV_trajectories[idx]

        if np.isscalar(val):
            x[...] = complex(val)
            continue

        arr = np.asarray(val)

        if arr.ndim == 0:
            x[...] = complex(arr)
            continue

        x[...] = complex(arr.flatten()[0])

    return result
