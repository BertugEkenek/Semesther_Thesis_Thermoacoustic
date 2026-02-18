import numpy as np
from utils import logger

# ============================================================
#              LINEAR FIRST-ORDER μ FIT
# ============================================================

# ------------------------------------------------------------
# Small (2×2) block for one n-sample
# ------------------------------------------------------------
def create_A_small(i, tau, w, sigma, s_ref: complex):

    sref2 = s_ref * s_ref
    denom = sref2.real**2 + sref2.imag**2

    s2 = (sigma[i] + 1j*w[i])**2

    C2 = (s2.real * sref2.real + s2.imag * sref2.imag) / denom
    C3 = (s2.imag * sref2.real - s2.real * sref2.imag) / denom

    s = np.sin(w[i] * tau)
    c = np.cos(w[i] * tau)

    return np.array([
        [C2*c + C3*s,  C2*s - C3*c],
        [C3*c - C2*s,  C3*s + C2*c]
    ], dtype=float)


# ------------------------------------------------------------
# RHS for one n-sample
# ------------------------------------------------------------
def create_b_small(config, w, sigma, s_ref, n, tau, s_k):
    sref2 = s_ref * s_ref
    denom = sref2.real**2 + sref2.imag**2

    # current normalized s^2 / s_ref^2
    s2 = (sigma + 1j * w) ** 2
    C2 = (s2.real * sref2.real + s2.imag * sref2.imag) / denom
    C3 = (s2.imag * sref2.real - s2.real * sref2.imag) / denom

    # initial-condition normalized s_k^2 / s_ref^2
    sk2 = s_k * s_k
    C4 = (sk2.real * sref2.real + sk2.imag * sref2.imag) / denom
    C5 = (sk2.imag * sref2.real - sk2.real * sref2.imag) / denom

    # forcing coefficient
    C1 = (
        config.Lambda[0]
        * config.alpha[0]
        * config.K
        * config.nu[0]
        * n
        * np.exp(-sigma * tau)
    )

    return np.array([(C2 - C4) / C1, (C3 - C5) / C1], dtype=float)



# ------------------------------------------------------------
# Vectorized A for one trajectory
# ------------------------------------------------------------
def create_A_column_trajectory_vectorized(n, tau, w, sigma, s_ref):
    # print("s_ref =", s_ref)

    w = np.asarray(w).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)

    m = len(n) - 1
    i = np.arange(m)

    sref2 = s_ref * s_ref
    denom = sref2.real**2 + sref2.imag**2

    s2 = (sigma[i] + 1j*w[i])**2

    C2 = (s2.real * sref2.real + s2.imag * sref2.imag) / denom
    C3 = (s2.imag * sref2.real - s2.real * sref2.imag) / denom

    s = np.sin(w[i] * tau)
    c = np.cos(w[i] * tau)

    A = np.zeros((2*m, 2), dtype=float)

    A[0::2, 0] = C2*c + C3*s
    A[0::2, 1] = C2*s - C3*c
    A[1::2, 0] = C3*c - C2*s
    A[1::2, 1] = C3*s + C2*c

    return A


# ------------------------------------------------------------
# Vectorized b for one trajectory
# ------------------------------------------------------------
def create_b_column_trajectory_vectorized(config, n, tau, w, sigma, s_ref):
    # print("s_ref =", s_ref)

    w = np.asarray(w).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)
    n = np.asarray(n).reshape(-1)

    m = len(n) - 1
    i = np.arange(m)

    sref2 = s_ref * s_ref
    denom = sref2.real**2 + sref2.imag**2

    s2 = (sigma + 1j*w)**2

    C2 = (s2.real * sref2.real + s2.imag * sref2.imag) / denom
    C3 = (s2.imag * sref2.real - s2.real * sref2.imag) / denom

    # reference-dependent constants
    s_k = s_ref
    # initial-condition normalized s_k^2 / s_ref^2
    sk2 = s_k * s_k
    C4 = (sk2.real * sref2.real + sk2.imag * sref2.imag) / denom
    C5 = (sk2.imag * sref2.real - sk2.real * sref2.imag) / denom

    
    C1 = (
        config.Lambda[0]
        * config.alpha[0]
        * config.K
        * config.nu[0]
        * n[1:]
        * np.exp(-sigma * tau)
    )
    

    b = np.zeros(2*m, dtype=float)
    b[0::2] = (-C4 + C2[i]) / C1
    b[1::2] = (C3[i] - C5) / C1
    # print("min|C1| =", np.min(np.abs(C1)))
    # print("max|b|  =", np.max(np.abs(b)))


    return b


# ------------------------------------------------------------
# Public builders (stack-safe)
# ------------------------------------------------------------
def build_A(n, tau, w_big, sigma_big, s_ref, use_only_acoustic=False, weights=None):

    if w_big.shape != sigma_big.shape:
        raise ValueError("w_big and sigma_big must have same shape")

    if use_only_acoustic:
        A = create_A_column_trajectory_vectorized(
            n, tau, w_big, sigma_big, s_ref
        )
        return A

    m, T = w_big.shape
    A_rows = []

    for j in range(T):
        Aj = create_A_column_trajectory_vectorized(
            n, tau, w_big[:, j], sigma_big[:, j], s_ref
        )
        if weights is not None:
            Aj *= weights[j]
        A_rows.append(Aj)

    return np.vstack(A_rows)


def build_b(config, n, tau, w_big, sigma_big, s_ref, use_only_acoustic=False, weights=None):

    if w_big.shape != sigma_big.shape:
        raise ValueError("w_big and sigma_big must have same shape")

    if use_only_acoustic:
        return create_b_column_trajectory_vectorized(
            config, n, tau, w_big, sigma_big, s_ref
        )

    m, T = w_big.shape
    b_rows = []

    for j in range(T):
        bj = create_b_column_trajectory_vectorized(
            config, n, tau, w_big[:, j], sigma_big[:, j], s_ref
        )
        if weights is not None:
            bj *= weights[j]
        b_rows.append(bj)

    return np.concatenate(b_rows)


# ------------------------------------------------------------
# LSQ solver
# ------------------------------------------------------------
def regression(
    A,
    b,
    check_condition_number: bool = True,
    quiet: bool = True,
    rcond=None,
):
    """
    Solve min ||A mu - b||_2 and log extended diagnostics.

    Baseline (uncorrected) comparison uses mu = [1, 0]
    which corresponds to μ = 1 + 0i.
    """

    A = np.asarray(A, float)
    b = np.asarray(b, float)

    if A.shape[1] != 2:
        raise ValueError(
            f"Linear μ-fit expects A with 2 columns (Re/Im). Got A.shape={A.shape}"
        )

    # ------------------------------------------------
    # Solve least squares
    # ------------------------------------------------
    mu, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=rcond)

    # ------------------------------------------------
    # Residuals
    # ------------------------------------------------
    r_fit = A @ mu - b

    mu_uncorrected = np.array([1.0, 0.0])  # μ = 1 + 0i
    r_uncorrected = A @ mu_uncorrected - b

    # ------------------------------------------------
    # Norms
    # ------------------------------------------------
    A_norm = np.linalg.norm(A, ord="fro")
    b_norm = np.linalg.norm(b)
    mu_norm = np.linalg.norm(mu)

    r_fit_norm = np.linalg.norm(r_fit)
    r_unc_norm = np.linalg.norm(r_uncorrected)

    rel_res_b = r_fit_norm / b_norm if b_norm > 0 else np.nan
    rel_res_scaled = (
        r_fit_norm / (A_norm * mu_norm + b_norm)
        if (A_norm * mu_norm + b_norm) > 0
        else np.nan
    )

    # ------------------------------------------------
    # Improvement metrics
    # ------------------------------------------------
    improvement_ratio = (
        r_fit_norm / r_unc_norm if r_unc_norm > 0 else np.nan
    )

    relative_reduction = (
        (r_unc_norm - r_fit_norm) / r_unc_norm
        if r_unc_norm > 0 else np.nan
    )

    # ------------------------------------------------
    # Conditioning diagnostics
    # ------------------------------------------------
    cond_A = np.linalg.cond(A) if check_condition_number else np.nan

    if svals is not None and len(svals) > 0:
        sigma_ratio = float(np.min(svals) / np.max(svals))
    else:
        sigma_ratio = np.nan

    # ------------------------------------------------
    # Logging
    # ------------------------------------------------
    logger.info("------ OLS Diagnostics ------")
    logger.info(f"||A||_F                = {A_norm:.3e}")
    logger.info(f"||b||_2                = {b_norm:.3e}")
    logger.info(f"||μ||_2                = {mu_norm:.3e}")

    logger.info(f"||r_fit||_2            = {r_fit_norm:.3e}")
    logger.info(f"||r_uncorrected||_2    = {r_unc_norm:.3e}")

    logger.info(f"rel_res (b)            = {rel_res_b:.3e}")
    logger.info(f"rel_res (scaled)       = {rel_res_scaled:.3e}")

    logger.info(f"improvement_ratio      = {improvement_ratio:.3e}")
    logger.info(f"relative_reduction     = {relative_reduction:.3e}")

    logger.info(f"rank                   = {rank}/{min(A.shape)}")
    logger.info(f"cond(A)                = {cond_A:.3e}")
    logger.info(f"σ_min/σ_max            = {sigma_ratio:.3e}")
    logger.info(f"mu (Re, Im)            = ({mu[0]:.3e}, {mu[1]:.3e})")
    logger.info("-----------------------------")

    return mu
