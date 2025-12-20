import numpy as np
from utils import logger

# ============================================================
#              LINEAR FIRST-ORDER μ FIT
#        (REFERENCE-NORMALIZED, MULTI-TRAJECTORY SAFE)
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
def create_b_small(config, w, sigma, s_ref, n, tau):

    sref2 = s_ref * s_ref
    denom = sref2.real**2 + sref2.imag**2

    s2 = (sigma + 1j*w)**2

    C2 = (s2.real * sref2.real + s2.imag * sref2.imag) / denom
    C3 = (s2.imag * sref2.real - s2.real * sref2.imag) / denom

    # forcing coefficient
    C1 = (
        config.Lambda[0]
        * config.alpha[0]
        * config.K
        * config.nu[0]
        * n
        * np.exp(-sigma * tau)
    )

    # reference-dependent constants
    sr, si = s_ref.real, s_ref.imag
    C4 = (si**2 - sr**2) * sref2.real / denom
    C5 = 2 * sr * si * sref2.real / denom

    return np.array([
        (C4 + C2) / C1,
        (C3 - C5) / C1
    ], dtype=float)


# ------------------------------------------------------------
# Vectorized A for one trajectory
# ------------------------------------------------------------
def create_A_column_trajectory_vectorized(n, tau, w, sigma, s_ref):


    w = np.asarray(w).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)

    m = len(n) - 1
    i = np.arange(m)

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

    sr, si = s_ref.real, s_ref.imag
    C4 = (si**2 - sr**2) * sref2.real / denom
    C5 = 2 * sr * si * sref2.real / denom

    C1 = (
        config.Lambda[0]
        * config.alpha[0]
        * config.K
        * config.nu[0]
        * n[1:]
        * np.exp(-sigma * tau)
    )

    b = np.zeros(2*m, dtype=float)
    b[0::2] = (C4 + C2[i]) / C1
    b[1::2] = (C3[i] - C5) / C1

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
def regression(A, b, check_condition_number=True, quiet=True, rcond=None):

    A = np.asarray(A, float)
    b = np.asarray(b, float)

    if check_condition_number and not quiet:
        logger.info(f"cond(A) = {np.linalg.cond(A):.2e}")
        logger.info(f"cond(b) = {np.linalg.cond(b.reshape(-1,1)):.2e}")

    mu, _, rank, _ = np.linalg.lstsq(A, b, rcond=rcond)

    if not quiet:
        logger.info(
            f"μ = {mu}, rank = {rank}, residual = {np.linalg.norm(A@mu - b):.3e}"
        )

    return mu
