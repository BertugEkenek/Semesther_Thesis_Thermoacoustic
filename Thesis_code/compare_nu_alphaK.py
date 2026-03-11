# compare_nu_alphaK.py

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def _mat_struct_to_dict(obj):
    if hasattr(obj, "_fieldnames"):
        return {name: _mat_struct_to_dict(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.size == 1:
            return _mat_struct_to_dict(obj.item())
        return [_mat_struct_to_dict(x) for x in obj.ravel()]
    return obj


def load_nu_from_mat(mat_path: str, config_name: str):
    """
    Returns dict keyed by negative R values to match your Python pipeline R-grid:
        out[-1.0] = {"nu11":..., "nu22":..., "nu12":..., "nu21":...}
    """
    mat = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    nu_data = _mat_struct_to_dict(mat["Nu_data"][0, 0])

    # Mapping from your config names to MAT names
    name_map = {
        "Rijke_tube_1": "Rijke1",
        "Rijke_tube_2": "Rijke2",
        "BRS": "BRS",
    }
    key = name_map[config_name]
    block = nu_data[key]

    out = {}
    for mag_key, vals in block.items():
        R_mag = float(np.asarray(vals["R_val"]).squeeze())
        R_py = -R_mag   # your pipeline uses negative R values

        out[R_py] = {
            "nu11": complex(np.asarray(vals["nu11"]).squeeze()),
            "nu22": complex(np.asarray(vals["nu22"]).squeeze()),
            "nu12": complex(np.asarray(vals["nu12"]).squeeze()),
            "nu21": complex(np.asarray(vals["nu21"]).squeeze()),
        }
    return out


def compute_alphaK_prefactors(config):
    """
    Same placement as in characteristic matrix:
      11 -> alpha1*K
      22 -> alpha2*K
      12 -> alpha2*K
      21 -> alpha1*K
    """
    a1K = config.alpha[0] * config.K
    a2K = config.alpha[1] * config.K

    return {
        "11": a1K,
        "22": a2K,
        "12": a2K,
        "21": a1K,
    }


def compute_reference_alphaK_nu(config, R_array, mat_nu_dict, atol=1e-10):
    """
    Returns complex arrays for alpha*K*nu_mat on the same R grid as your fit.

    Uses tolerant matching because values like -0.8 and -0.7999999999999999
    should be treated as the same R.
    """
    pref = compute_alphaK_prefactors(config)

    available_R = np.array(sorted(mat_nu_dict.keys()), dtype=float)

    ref11, ref22, ref12, ref21 = [], [], [], []

    for R in np.asarray(R_array, dtype=float):
        idx = np.where(np.isclose(available_R, R, atol=atol, rtol=0.0))[0]

        if len(idx) == 0:
            raise KeyError(
                f"R={R} not found in MAT nu data within atol={atol}. "
                f"Available: {available_R.tolist()}"
            )

        if len(idx) > 1:
            raise ValueError(
                f"Ambiguous tolerant match for R={R}. Matches: {available_R[idx].tolist()}"
            )

        R_match = float(available_R[idx[0]])
        nuR = mat_nu_dict[R_match]

        ref11.append(pref["11"] * nuR["nu11"])
        ref22.append(pref["22"] * nuR["nu22"])
        ref12.append(pref["12"] * nuR["nu12"])
        ref21.append(pref["21"] * nuR["nu21"])

    return {
        "11": np.asarray(ref11, dtype=complex),
        "22": np.asarray(ref22, dtype=complex),
        "12": np.asarray(ref12, dtype=complex),
        "21": np.asarray(ref21, dtype=complex),
    }
def compute_fitted_alphaK_mu_nu(config, mu_array):
    """
    Build alpha*K*(mu*nu) directly from mu_array shape.

    Accepted layouts
    ----------------
    mu_array.shape[1] == 6:
        [Re11, Im11, Re22, Im22, Re12, Im12]
        -> mu21 = mu12

    mu_array.shape[1] == 8:
        [Re11, Im11, Re22, Im22, Re12, Im12, Re21, Im21]
    """
    mu_array = np.asarray(mu_array, dtype=float)

    if mu_array.ndim != 2:
        raise ValueError("mu_array must be 2D.")

    _, p_dim = mu_array.shape
    if p_dim not in (6, 8):
        raise ValueError(f"mu_array must have 6 or 8 columns, got {p_dim}.")

    pref = compute_alphaK_prefactors(config)

    mu11 = mu_array[:, 0] + 1j * mu_array[:, 1]
    mu22 = mu_array[:, 2] + 1j * mu_array[:, 3]
    mu12 = mu_array[:, 4] + 1j * mu_array[:, 5]

    if p_dim == 6:
        mu21 = mu12
    else:
        mu21 = mu_array[:, 6] + 1j * mu_array[:, 7]

    return {
        "11": pref["11"] * (mu11 * config.nu[0]),
        "22": pref["22"] * (mu22 * config.nu[1]),
        "12": pref["12"] * (mu12 * config.nu[2]),
        "21": pref["21"] * (mu21 * config.nu[3]),
    }
def plot_alphaK_comparison_complex_plane(R, ref_eff, fit_eff, title_suffix=""):
    labels = ["11", "22", "12", "21"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.ravel()

    cmap = plt.cm.viridis
    norm = plt.Normalize(np.min(R), np.max(R))

    for ax, key in zip(axes, labels):
        sc1 = ax.scatter(
            ref_eff[key].real,
            ref_eff[key].imag,
            c=R,
            cmap=cmap,
            norm=norm,
            marker="o",
            s=60,
            label=r"$\alpha K \nu$ (.mat)",
        )
        ax.scatter(
            fit_eff[key].real,
            fit_eff[key].imag,
            c=R,
            cmap=cmap,
            norm=norm,
            marker="x",
            s=70,
            label=r"$\alpha K \mu \nu$ (fit)",
        )

        for i, r in enumerate(R):
            ax.annotate(f"{r:.2f}", (ref_eff[key].real[i], ref_eff[key].imag[i]), fontsize=8, alpha=0.7)
            ax.annotate(f"{r:.2f}", (fit_eff[key].real[i], fit_eff[key].imag[i]), fontsize=8, alpha=0.7)

        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title(f"Component {key}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.legend()

    cbar = fig.colorbar(sc1, ax=axes, shrink=0.9)
    cbar.set_label("R")
    fig.suptitle(f"Comparison in complex plane: αKν vs αKμν {title_suffix}", fontsize=14)
    return fig, axes

def plot_alphaK_comparison_complex_plane_separate(
    R,
    ref_eff,
    fit_eff,
    title_suffix="",
    show_labels=True,
    connect_pairs=True,
):
    import numpy as np
    import matplotlib.pyplot as plt

    labels = ["11", "22", "12", "21"]
    figs_axes = {}

    R = np.asarray(R, dtype=float)
    cmap = plt.cm.viridis
    norm = plt.Normalize(np.min(R), np.max(R))

    for key in labels:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

        # reference points
        sc = ax.scatter(
            ref_eff[key].real,
            ref_eff[key].imag,
            c=R,
            cmap=cmap,
            norm=norm,
            marker="o",
            s=70,
            label=r"$\alpha K \nu$ (.mat)",
        )

        # fitted points
        ax.scatter(
            fit_eff[key].real,
            fit_eff[key].imag,
            c=R,
            cmap=cmap,
            norm=norm,
            marker="x",
            s=80,
            label=r"$\alpha K \mu \nu$ (fit)",
        )

        # optional pair connections
        if connect_pairs:
            for i in range(len(R)):
                ax.plot(
                    [ref_eff[key].real[i], fit_eff[key].real[i]],
                    [ref_eff[key].imag[i], fit_eff[key].imag[i]],
                    linewidth=0.8,
                    alpha=0.7,
                )

        # optional labels: only annotate reference points
        if show_labels:
            for i, r in enumerate(R):
                ax.annotate(
                    f"{r:.2f}",
                    (ref_eff[key].real[i], ref_eff[key].imag[i]),
                    fontsize=9,
                    alpha=0.8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )

        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title(f"Component {key} {title_suffix}")
        ax.grid(True)
        ax.legend()

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("R")

        figs_axes[key] = (fig, ax)

    return figs_axes