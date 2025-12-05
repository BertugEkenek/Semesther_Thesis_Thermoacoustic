import numpy as np
from utils import logger

# ---------------- LINEAR FIT FUNCTIONS ---------------- #

def create_A_small(i: int, tau: float, w: np.ndarray, sigma: np.ndarray, sigma0=0.0, w0=0.0):
    denom = (sigma0**2 - w0**2)**2 + (2 * sigma0 * w0)**2
    C2 = ((sigma[i]**2 - w[i]**2) * (sigma0**2 - w0**2) + 4 * sigma[i] * w[i] * sigma0 * w0) / denom
    C3 = (2 * sigma[i] * w[i] * (sigma0**2 - w0**2) - (sigma[i]**2 - w[i]**2) * sigma0 * w0) / denom

    s = np.sin(w[i] * tau)
    c = np.cos(w[i] * tau)

    return np.array([
        [C2 * c + C3 * s,  C2 * s - C3 * c],
        [C3 * c - C2 * s,  C3 * s + C2 * c]
    ], dtype=float)


def create_b_small(config: object, w: np.ndarray, sigma: np.ndarray, s_1: complex, n: float, tau: float, sigma0: float, w0: float):
    C1 = config.Lambda[0] * config.alpha[0] * config.K * config.nu[0] * n * np.exp(-sigma * tau)

    denom = (sigma0**2 - w0**2)**2 + (2 * sigma0 * w0)**2
    C2 = ((sigma**2 - w**2) * (sigma0**2 - w0**2) + 4 * sigma * w * sigma0 * w0) / denom
    C3 = (2 * sigma * w * (sigma0**2 - w0**2) - (sigma**2 - w**2) * sigma0 * w0) / denom

    w1, sigma1 = s_1.imag, s_1.real
    C4 = ((w1**2 - sigma1**2) * (sigma0**2 - w0**2) - 4 * sigma1 * w1 * sigma0 * w0) / denom
    C5 = 2 * (sigma1 * w1 * (sigma0**2 - w0**2) + (-w1**2 + sigma1**2) * sigma0 * w0) / denom

    return np.array([(C4 + C2) / C1, (C3 - C5) / C1], dtype=float)


def create_b_column_trajectory(config: object, n: np.ndarray, tau: float, w: np.ndarray, sigma: np.ndarray, s_1: complex, sigma0: float, w0: float):
    m = len(n) - 1
    b = np.zeros(2 * m, dtype=float)
    for i in range(m):
        b[2 * i:2 * i + 2] = create_b_small(config, w[i], sigma[i], s_1, n[i + 1], tau, sigma0, w0)
    return b


def create_A_column_trajectory(n: np.ndarray, tau: float, w: np.ndarray, sigma: np.ndarray, sigma0: float, w0: float):
    m = len(n) - 1
    A = np.zeros((2 * m, 2), dtype=float)
    for i in range(m):
        A[2 * i:2 * i + 2, :] = create_A_small(i, tau, w, sigma, sigma0, w0)
    return A


def build_A(n: np.ndarray, tau: float, w_big: np.ndarray, sigma_big: np.ndarray, sigma0: float, w0: float, use_only_acoustic=False, weights=None):
    if w_big.shape != sigma_big.shape :
        raise ValueError("w_big and sigma_big must have same shape (m, T).")
    if use_only_acoustic:
        m = w_big.shape[0]
        T = 1
    else: 
        m, T = w_big.shape
    if len(n) - 1 != m:
        raise ValueError(f"len(n)-1 must equal m. Got {len(n)-1} vs {m}.")
    if weights is not None and weights.shape != (T,):
        raise ValueError(f"weights must have shape (T,), got {weights.shape}.")

    A_rows = []
    if use_only_acoustic:
        A_j = create_A_column_trajectory_vectorized(n, tau, w_big, sigma_big, sigma0, w0)
        if weights is not None:
            A_j *= weights[0]
        A_rows.append(A_j)
        return np.vstack(A_rows)
    for j in range(T):
        A_j = create_A_column_trajectory_vectorized(n, tau, w_big[:, j], sigma_big[:, j],sigma0, w0)
        if weights is not None:
            A_j *= weights[j]
        A_rows.append(A_j)
    
    return np.vstack(A_rows)


def build_b(config: object, n: np.ndarray, tau: float, w_big: np.ndarray, sigma_big: np.ndarray, s_1: complex, sigma0: float, w0: float,use_only_acoustic=False, weights=None):
    if w_big.shape != sigma_big.shape:
        raise ValueError("w_big and sigma_big must have same shape (m, T).")
    if use_only_acoustic:
        m = w_big.shape[0]
        T = 1
    else: 
        m, T = w_big.shape
    if len(n) - 1 != m:
        raise ValueError(f"len(n)-1 must equal m. Got {len(n)-1} vs {m}.")
    if weights is not None and weights.shape != (T,):
        raise ValueError(f"weights must have shape (T,), got {weights.shape}.")
    if use_only_acoustic:
        b_j = create_b_column_trajectory_vectorized(config, n, tau,  w_big, sigma_big, s_1, sigma0, w0)
        if weights is not None:
            b_j *= weights[0]
        return b_j
    b_rows = []
    for j in range(T):
        b_j = create_b_column_trajectory_vectorized(config, n,tau, w_big[:, j], sigma_big[:, j], s_1, sigma0, w0)
        if weights is not None:
            b_j *= weights[j]
        b_rows.append(b_j)
    return np.concatenate(b_rows)


def regression(A: np.ndarray, b: np.ndarray, check_condition_number=True, quiet=True, rcond=None):
    """Solve A μ ≈ b."""
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    if check_condition_number and not quiet:
        logger.info(f"cond(A): {np.linalg.cond(A)}")
    mu, _, rank, _ = np.linalg.lstsq(A, b, rcond=rcond)
    if not quiet:
        logger.info(f"μ: {mu}, rank: {rank}, residual: {np.linalg.norm(A @ mu - b)}")
    return mu

def create_A_column_trajectory_vectorized(n: np.ndarray,
                                          tau: float,
                                          w: np.ndarray,
                                          sigma: np.ndarray,
                                          sigma0: float,
                                          w0: float):
    """
    Vectorized equivalent of create_A_column_trajectory().
    Produces identical numerical output, drastically faster.
    """

    m = len(n) - 1                     # matches loop version
    i = np.arange(m)                   # index array

    # ----- Compute constants -----
    denom = (sigma0**2 - w0**2)**2 + (2 * sigma0 * w0)**2

    C2 = ((sigma[i]**2 - w[i]**2) * (sigma0**2 - w0**2) +
          4 * sigma[i] * w[i] * sigma0 * w0) / denom

    C3 = (2 * sigma[i] * w[i] * (sigma0**2 - w0**2) -
          (sigma[i]**2 - w[i]**2) * sigma0 * w0) / denom

    s = np.sin(w[i] * tau)
    c = np.cos(w[i] * tau)

    # ----- Build blocks (vectorized) -----
    A11 = C2 * c + C3 * s
    A12 = C2 * s - C3 * c
    A21 = C3 * c - C2 * s
    A22 = C3 * s + C2 * c

    # ----- Assemble into final A matrix (2m × 2) -----
    A = np.zeros((2*m, 2), dtype=float)

    A[0::2, 0] = A11
    A[0::2, 1] = A12
    A[1::2, 0] = A21
    A[1::2, 1] = A22

    return A
def create_b_column_trajectory_vectorized(config: object,
                                          n: np.ndarray,
                                          tau: float,
                                          w: np.ndarray,
                                          sigma: np.ndarray,
                                          s_1: complex,
                                          sigma0: float,
                                          w0: float):

    m = len(n) - 1
    i = np.arange(m)

    # ----- Precompute constants -----
    C1 = (
        config.Lambda[0]
        * config.alpha[0]
        * config.K
        * config.nu[0]
        * n[1:]                      # n[i+1] from original code
        * np.exp(-sigma * tau)
    )

    denom = (sigma0**2 - w0**2)**2 + (2 * sigma0 * w0)**2

    # ----- C2, C3 -----
    C2 = ((sigma**2 - w**2) * (sigma0**2 - w0**2) +
          4 * sigma * w * sigma0 * w0) / denom

    C3 = (2 * sigma * w * (sigma0**2 - w0**2) -
          (sigma**2 - w**2) * sigma0 * w0) / denom

    # ----- C4, C5 depend only on s_1 -----
    w1 = s_1.imag
    sigma1 = s_1.real

    C4 = ((w1**2 - sigma1**2) * (sigma0**2 - w0**2) -
          4 * sigma1 * w1 * sigma0 * w0) / denom

    C5 = 2 * (
        sigma1 * w1 * (sigma0**2 - w0**2) +
        (-w1**2 + sigma1**2) * sigma0 * w0
    ) / denom

    # ----- b1, b2 -----
    b1 = (C4 + C2[i]) / C1
    b2 = (C3[i] - C5) / C1

    # ----- Final b vector -----
    b = np.zeros(2*m, dtype=float)
    b[0::2] = b1
    b[1::2] = b2

    return b
