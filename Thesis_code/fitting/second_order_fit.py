import numpy as np
from scipy.optimize import least_squares
from utils import logger

# ---------------- SECOND-ORDER (NONLINEAR) FIT ---------------- #

def create_M11M22minusM12M21_small(config: object ,i : int, tau: float,
                                   w: np.ndarray, sigma: np.ndarray,
                                   s_1: complex ,s_2: complex, n: float) -> tuple[np.ndarray, np.ndarray]:

    sigma1 = s_1.real
    sigma2 = s_2.real
    w1 = s_1.imag
    w2 = s_2.imag
    alphaK1 = config.alpha[0] * config.K
    alphaK2 = config.alpha[1] * config.K

    Cr1 = sigma1**2 - w1**2
    Ci1 = 2 * sigma1 * w1
    Cr2 = sigma2**2 - w2**2
    Ci2 = 2 * sigma2 * w2
    
    denom1 = Cr1**2 - Ci1**2

    # Diagnostics
    eps = 1e-6  # or smaller, depending on your scales
    if abs(denom1) < eps:
        logger.warning(
            f"[Singular warning] denom1 ~ 0 at i={i}, n={n}, "
            f"s1={s_1}, s2={s_2}, Cr1={Cr1}, Ci1={Ci1}, denom1={denom1}"
        )

    C11 = config.Lambda[0] * alphaK1 * config.nu[0] * n * np.exp(-tau*sigma[i])
    C22 = config.Lambda[1] * alphaK2 * config.nu[1] * n * np.exp(-tau*sigma[i])
    C12 = config.Lambda[1] * alphaK1 * config.nu[2] * n * np.exp(-tau*sigma[i])
    C21 = config.Lambda[0] * alphaK2 * config.nu[3] * n * np.exp(-tau*sigma[i])

    if abs(C11) < 1e-10 and abs(C22) < 1e-10 and abs(C12) < 1e-10 and abs(C21) < 1e-10:
        logger.warning(f"All Cij very small at i={i}, n={n}, sigma={sigma[i]}, w={w[i]}")

    Cr = sigma[i]**2 - w[i]**2
    Ci = 2 * sigma[i] * w[i]

    Crn1 = (Cr * Cr1 + Ci * Ci1)/(Cr1**2 - Ci1**2)
    Cin1 = (Cr1 * Ci - Cr * Ci1)/(Cr1**2 - Ci1**2)

    Cr2n1 = (Cr2 * Cr1 + Ci2 * Ci1)/(Cr1**2 - Ci1**2)
    Ci2n1 = (Cr2 * Ci2 - Ci2 * Ci1)/(Cr1**2 - Ci1**2)

    s = np.sin(w[i] * tau)
    c = np.cos(w[i] * tau)

    block_M11M22_M12M21 = np.array([
        [
            C11*(Cin1**2*c - Crn1**2*c - Ci2n1*Cin1*c + Cr2n1*Crn1*c +
                 Ci2n1*Crn1*s + Cin1*Cr2n1*s - 2*Cin1*Crn1*s),
            -C11*(Crn1**2*s - Cin1**2*s + Ci2n1*Crn1*c + Cin1*Cr2n1*c -
                  2*Cin1*Crn1*c + Ci2n1*Cin1*s - Cr2n1*Crn1*s),
            C22*(Crn1*c + Cin1*s + Cin1**2*c - Crn1**2*c - 2*Cin1*Crn1*s),
            C22*(Crn1*s - Cin1*c + Cin1**2*s - Crn1**2*s + 2*Cin1*Crn1*c),
            C11*C22*(Crn1**2*c**2 - Cin1**2*c**2 + Cin1**2*s**2 -
                     Crn1**2*s**2 + 4*Cin1*Crn1*c*s),
            -2*C11*C22*(Cin1*Crn1*c**2 - Cin1*Crn1*s**2 +
                        Cin1**2*c*s - Crn1**2*c*s),
            -2*C11*C22*(Cin1*Crn1*c**2 - Cin1*Crn1*s**2 +
                        Cin1**2*c*s - Crn1**2*c*s),
            -C11*C22*(Crn1**2*c**2 - Cin1**2*c**2 + Cin1**2*s**2 -
                      Crn1**2*s**2 + 4*Cin1*Crn1*c*s),
            -C12*C21*(Crn1**2*c**2 - Cin1**2*c**2 + Cin1**2*s**2 -
                      Crn1**2*s**2 + 4*Cin1*Crn1*c*s),
            2*C12*C21*(Cin1*Crn1*c**2 - Cin1*Crn1*s**2 +
                       Cin1**2*c*s - Crn1**2*c*s),
            2*C12*C21*(Cin1*Crn1*c**2 - Cin1*Crn1*s**2 +
                       Cin1**2*c*s - Crn1**2*c*s),
            C12*C21*(Crn1**2*c**2 - Cin1**2*c**2 + Cin1**2*s**2 -
                     Crn1**2*s**2 + 4*Cin1*Crn1*c*s),
        ],
        [
            C11*(Crn1**2*s - Cin1**2*s + Ci2n1*Crn1*c +
                 Cin1*Cr2n1*c - 2*Cin1*Crn1*c + Ci2n1*Cin1*s -
                 Cr2n1*Crn1*s),
            C11*(Cin1**2*c - Crn1**2*c - Ci2n1*Cin1*c +
                 Cr2n1*Crn1*c + Ci2n1*Crn1*s + Cin1*Cr2n1*s -
                 2*Cin1*Crn1*s),
            -C22*(Crn1*s - Cin1*c + Cin1**2*s - Crn1**2*s +
                  2*Cin1*Crn1*c),
            C22*(Crn1*c + Cin1*s + Cin1**2*c - Crn1**2*c -
                 2*Cin1*Crn1*s),
            2*C11*C22*(Cin1*Crn1*c**2 - Cin1*Crn1*s**2 +
                       Cin1**2*c*s - Crn1**2*c*s),
            C11*C22*(Crn1**2*c**2 - Cin1**2*c**2 + Cin1**2*s**2 -
                     Crn1**2*s**2 + 4*Cin1*Crn1*c*s),
            C11*C22*(Crn1**2*c**2 - Cin1**2*c**2 + Cin1**2*s**2 -
                     Crn1**2*s**2 + 4*Cin1*Crn1*c*s),
            -2*C11*C22*(Cin1*Crn1*c**2 - Cin1*Crn1*s**2 +
                        Cin1**2*c*s - Crn1**2*c*s),
            -2*C12*C21*(Cin1*Crn1*c**2 - Cin1*Crn1*s**2 +
                        Cin1**2*c*s - Crn1**2*c*s),
            -C12*C21*(Crn1**2*c**2 - Cin1**2*c**2 + Cin1**2*s**2 -
                      Crn1**2*s**2 + 4*Cin1*Crn1*c*s),
            -C12*C21*(Crn1**2*c**2 - Cin1**2*c**2 + Cin1**2*s**2 -
                      Crn1**2*s**2 + 4*Cin1*Crn1*c*s),
            2*C12*C21*(Cin1*Crn1*c**2 - Cin1*Crn1*s**2 +
                       Cin1**2*c*s - Crn1**2*c*s),
        ],
    ], dtype=float)

    b_vector = np.array([
        -(Cr2n1 - Crn1 - Cin1**2 + Crn1**2 + Ci2n1*Cin1 -
          Cr2n1*Crn1),
        -(Ci2n1 - Cin1 - Ci2n1*Crn1 - Cin1*Cr2n1 +
          2*Cin1*Crn1),
    ], dtype=float)

    return block_M11M22_M12M21, b_vector



# ---------- nonlinear residual ---------- #

def residual(p: np.ndarray, config: object, i: int, tau: float,
             w: np.ndarray, sigma: np.ndarray,
             s_1: complex, s_2: complex, n: float,
             enforce_symmetry: bool):

    mu = build_mu_from_p(p, enforce_symmetry)

    M, b = create_M11M22minusM12M21_small(
        config, i, tau, w, sigma, s_1, s_2, n
    )

    if np.isnan(M).any():
        print("M has NaN:", np.isnan(M).any(), "inf:", np.isinf(M).any())
        print("cond(M):", np.linalg.cond(M))
    if np.isnan(b).any():
        print("b has NaN:", np.isnan(b).any(), "inf:", np.isinf(b).any())
        print("b:", b)

    return M @ mu - b



# ---------- nonlinear trust-region solve ---------- #

def find_mu(config: object, i: int, tau: float, w: np.ndarray,
            sigma: np.ndarray, s_1: complex, s_2: complex, n: float,
            enforce_symmetry: bool):
    # Make the initialization random values between 0 and 2 maybe?
    if enforce_symmetry:
        p0 = np.ones(6)* np.random.random_sample()
        args = (config, i, tau, w, sigma, s_1, s_2, n, True)
    else:
        p0 = np.ones(8)* np.random.random_sample()
        args = (config, i, tau, w, sigma, s_1, s_2, n, False)

    res = least_squares(residual, p0, args=args, method="trf")
    if (res.success == False):
        print("success:", res.success, "status:", res.status, "cost:", res.cost)
        print("x (result):", res.x)
    x = res.x

    if enforce_symmetry:
        mur11, mui11, mur22, mui22, mur12, mui12 = x
        mur21, mui21 = mur12, mui12
    else:
        mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21 = x

    mu = np.array([
        mur11, mui11, mur22, mui22,
        mur11 * mur22, mur11 * mui22, mui11 * mur22, mui11 * mui22,
        mur12 * mur21, mur12 * mui21, mui12 * mur21, mui12 * mui21
    ])

    return mu, (mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21)


# ---------- nonlinear regression per sample ---------- #

def regression_nonlinear_per_sample(config: object, n: np.ndarray, tau: float,
                                    w_big: np.ndarray, sigma_big: np.ndarray,
                                    s_1: complex, s_2: complex,
                                    use_only_acoustic: bool,enforce_symmetry: bool,
                                    weights=None, method='trf', quiet=False,
                                    ):

    if use_only_acoustic:
        w_big = w_big[:, :1]
        sigma_big = sigma_big[:, :1]

    if w_big.shape != sigma_big.shape:
        raise ValueError("w_big and sigma_big must have same shape (m, T).")

    m, T = w_big.shape
    if len(n) - 1 != m:
        raise ValueError(f"len(n)-1 must equal m. Got {len(n)-1} vs {m}.")

    if isinstance(weights, bool):
        weights = None
    if weights is not None and weights.shape != (T,):
        raise ValueError(f"weights must have shape (T,), got {weights.shape}.")

    mu_array = np.zeros((m, T, 12), dtype=float)
    mu_value_array = np.zeros((m, T, 8), dtype=float)
    residual_norms = np.zeros((m, T), dtype=float)
    success = np.zeros((m, T), dtype=bool)

    for j in range(T):
        # Precompute all M, b for this column j (vectorized over i)
        col_w = w_big[:, j].copy()
        col_sigma = sigma_big[:, j].copy()

        M_all, b_all = create_M11M22minusM12M21_vectorized(
            config, tau, col_w, col_sigma, s_1, s_2, n
        )   # M_all: (m, 2, 12), b_all: (m, 2)

        for i in range(m):
            try:
                M_block = M_all[i, :, :]      # (2, 12)
                b_block = b_all[i, :]         # (2,)

                mu_vec, mvals = find_mu_from_block(
                    M_block, b_block, enforce_symmetry, quiet=quiet
                )

                mu_array[i, j, :] = mu_vec
                mu_value_array[i, j, :] = mvals

                res_vec = M_block @ mu_vec - b_block
                residual_norms[i, j] = np.linalg.norm(res_vec)
                success[i, j] = True

            except Exception as exc:
                if not quiet:
                    logger.error(f"find_mu_from_block failed at i={i}, j={j}: {exc}")
                mu_array[i, j, :] = np.nan
                mu_value_array[i, j, :] = np.nan
                residual_norms[i, j] = np.nan
                success[i, j] = False

    info = {
        "residual_norms": residual_norms,
        "success": success,
        "mu_values": mu_value_array
    }
    return mu_array, mu_value_array, info


# ---------- stacked 2D arrays + global regression ---------- #

def mu_array_stacked(config: object, n: np.ndarray, tau: float,
                     w_big: np.ndarray, sigma_big: np.ndarray,
                     s_1: complex, s_2: complex, use_only_acoustic: bool, enforce_symmetry: bool,
                     weights=None, quiet=False,
                     ):

    def ensure_2d(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr[:, None]
        return arr

    w_big = ensure_2d(w_big)
    sigma_big = ensure_2d(sigma_big)

    if use_only_acoustic:
        w_big = w_big[:, :1]
        sigma_big = sigma_big[:, :1]

    m, T = w_big.shape

    mu3d, mu_value_array, info = regression_nonlinear_per_sample(
        config, n, tau, w_big, sigma_big, s_1, s_2,
        use_only_acoustic,enforce_symmetry,
        weights=weights, quiet=quiet,
        
    )

    mu_stacked = mu3d.reshape((m*T, 12))

    # Precompute all M, b for all j using the vectorized builder
    M_list = []
    b_list = []
    for j in range(T):
        M_j, b_j = create_M11M22minusM12M21_vectorized(
            config, tau, w_big[:, j], sigma_big[:, j], s_1, s_2, n
        )
        M_list.append(M_j)   # (m, 2, 12)
        b_list.append(b_j)   # (m, 2)

    M_all_stacked = np.concatenate(M_list, axis=0)   # (m*T, 2, 12)
    b_all_stacked = np.concatenate(b_list, axis=0)   # (m*T, 2)

    def residual_global(p):
        """
        Vectorized global residual over all (i,j).
        """
        # Build μ from p
        mu_vec = build_mu_from_p(p, enforce_symmetry)   # (12,)

        # Core residual M μ - b for all samples
        # Result shape: (m*T, 2)
        R = M_all_stacked @ mu_vec - b_all_stacked

        # Flatten all 2-components into a single vector
        res = R.reshape(-1)

        # -------------------------------------
        # Regularization toward μ = 1 + 0i
        # -------------------------------------
        if enforce_symmetry:
            p_target = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        else:
            p_target = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        lam = 0.05  # Regularization strength
        reg = np.sqrt(lam) * (p - p_target)

        return np.concatenate([res, reg])



    if enforce_symmetry:
        p0 = np.ones(6) * np.random.random_sample()
    else:
        valid_mask = np.isfinite(mu_value_array).all(axis=2).any(axis=1)
        if np.any(valid_mask):
            p0 = np.nanmean(mu_value_array[valid_mask].reshape(-1, 8), axis=0)
        else:
            p0 = np.ones(8) *np.random.random_sample()
    
    # # --- Debug: check if residual actually depends on μ ---

    # # Choose some test μ’s
    # if enforce_symmetry:
    #     p_test_ones  = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])  # μ11=μ22=μ12=1+0i
    #     p_test_zeros = np.zeros_like(p_test_ones)                # μ=0
    #     p0_debug     = p0.copy()
    # else:
    #     p_test_ones  = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    #     p_test_zeros = np.zeros_like(p_test_ones)
    #     p0_debug     = p0.copy()

    # def norm_res(p):
    #     r = residual_global(p)
    #     return np.linalg.norm(r[~np.isnan(r)])

    # print("DEBUG μ-fit:")
    # print("  p0        =", p0_debug)
    # print("  ||res(p0)||   =", norm_res(p0_debug))
    # print("  ||res(μ=1)||  =", norm_res(p_test_ones))
    # print("  ||res(μ=0)||  =", norm_res(p_test_zeros))

    # r_p0    = residual_global(p0_debug)
    # r_ones  = residual_global(p_test_ones)
    # r_zeros = residual_global(p_test_zeros)

    # print("max |r_p0 - r_ones|  =", np.nanmax(np.abs(r_p0 - r_ones)))
    # print("max |r_p0 - r_zeros| =", np.nanmax(np.abs(r_p0 - r_zeros)))



    res = least_squares(residual_global, p0, method="trf")
    mu_opt = res.x

    info_stacked = {
        "residual_norms": info["residual_norms"].reshape((m*T,)),
        "success": info["success"].reshape((m*T,)),
        "global_cost": res.cost
    }

    return mu_stacked, mu_opt, info_stacked

def create_M11M22minusM12M21_vectorized(
        config: object,
        tau: float,
        w: np.ndarray,
        sigma: np.ndarray,
        s_1: complex,
        s_2: complex,
        n: np.ndarray):
    """
    Vectorized version of create_M11M22minusM12M21_small().
    For a given w, sigma column and full n, returns:

        M_all : (m, 2, 12)
        b_all : (m, 2)

    where m = len(n) - 1.
    """

    m = len(n) - 1
    idx = np.arange(m)

    # ---- mode data (s1, s2) ----
    sigma1, sigma2 = s_1.real, s_2.real
    w1, w2 = s_1.imag, s_2.imag

    alphaK1 = config.alpha[0] * config.K
    alphaK2 = config.alpha[1] * config.K

    Cr1 = sigma1**2 - w1**2
    Ci1 = 2 * sigma1 * w1
    Cr2 = sigma2**2 - w2**2
    Ci2 = 2 * sigma2 * w2

    denom1 = Cr1**2 - Ci1**2
    eps = 1e-6
    if abs(denom1) < eps:
        logger.warning(
            f"[Singular warning] denom1 ~ 0 in vectorized M/b computation, "
            f"s1={s_1}, s2={s_2}, Cr1={Cr1}, Ci1={Ci1}, denom1={denom1}"
        )

    # ---- C11, C22, C12, C21 (vectorized over i) ----
    exp_term = np.exp(-tau * sigma[idx])

    C11 = config.Lambda[0] * alphaK1 * config.nu[0] * n[1:] * exp_term
    C22 = config.Lambda[1] * alphaK2 * config.nu[1] * n[1:] * exp_term
    C12 = config.Lambda[1] * alphaK1 * config.nu[2] * n[1:] * exp_term
    C21 = config.Lambda[0] * alphaK2 * config.nu[3] * n[1:] * exp_term

    # ---- Cr, Ci (trajectory) ----
    Cr = sigma[idx]**2 - w[idx]**2
    Ci = 2 * sigma[idx] * w[idx]

    Crn1 = (Cr * Cr1 + Ci * Ci1) / denom1
    Cin1 = (Cr1 * Ci - Cr * Ci1) / denom1

    Cr2n1 = (Cr2 * Cr1 + Ci2 * Ci1) / denom1
    Ci2n1 = (Cr2 * Ci2 - Ci2 * Ci1) / denom1

    s = np.sin(w[idx] * tau)
    c = np.cos(w[idx] * tau)

    # ---- Allocate outputs ----
    M_all = np.zeros((m, 2, 12), dtype=float)
    b_all = np.zeros((m, 2), dtype=float)

    # ---------------- Row 0 ----------------
    M_all[:, 0, 0] = C11 * (Cin1**2 * c - Crn1**2 * c - Ci2n1 * Cin1 * c +
                            Cr2n1 * Crn1 * c + Ci2n1 * Crn1 * s +
                            Cin1 * Cr2n1 * s - 2 * Cin1 * Crn1 * s)

    M_all[:, 0, 1] = -C11 * (Crn1**2 * s - Cin1**2 * s + Ci2n1 * Crn1 * c +
                             Cin1 * Cr2n1 * c - 2 * Cin1 * Crn1 * c +
                             Ci2n1 * Cin1 * s - Cr2n1 * Crn1 * s)

    M_all[:, 0, 2] = C22 * (Crn1 * c + Cin1 * s + Cin1**2 * c -
                            Crn1**2 * c - 2 * Cin1 * Crn1 * s)

    M_all[:, 0, 3] = C22 * (Crn1 * s - Cin1 * c + Cin1**2 * s -
                            Crn1**2 * s + 2 * Cin1 * Crn1 * c)

    M_all[:, 0, 4] = C11 * C22 * (Crn1**2 * c**2 - Cin1**2 * c**2 +
                                  Cin1**2 * s**2 - Crn1**2 * s**2 +
                                  4 * Cin1 * Crn1 * c * s)

    M_all[:, 0, 5] = -2 * C11 * C22 * (Cin1 * Crn1 * c**2 -
                                       Cin1 * Crn1 * s**2 +
                                       Cin1**2 * c * s -
                                       Crn1**2 * c * s)

    M_all[:, 0, 6] = M_all[:, 0, 5]
    M_all[:, 0, 7] = -M_all[:, 0, 4]

    M_all[:, 0, 8] = -C12 * C21 * (Crn1**2 * c**2 - Cin1**2 * c**2 +
                                   Cin1**2 * s**2 - Crn1**2 * s**2 +
                                   4 * Cin1 * Crn1 * c * s)

    M_all[:, 0, 9] = 2 * C12 * C21 * (Cin1 * Crn1 * c**2 -
                                      Cin1 * Crn1 * s**2 +
                                      Cin1**2 * c * s -
                                      Crn1**2 * c * s)

    M_all[:, 0, 10] = M_all[:, 0, 9]
    M_all[:, 0, 11] = -M_all[:, 0, 8]

    # ---------------- Row 1 ----------------
    M_all[:, 1, 0] = C11 * (Crn1**2 * s - Cin1**2 * s + Ci2n1 * Crn1 * c +
                            Cin1 * Cr2n1 * c - 2 * Cin1 * Crn1 * c +
                            Ci2n1 * Cin1 * s - Cr2n1 * Crn1 * s)

    M_all[:, 1, 1] = C11 * (Cin1**2 * c - Crn1**2 * c - Ci2n1 * Cin1 * c +
                            Cr2n1 * Crn1 * c + Ci2n1 * Crn1 * s +
                            Cin1 * Cr2n1 * s - 2 * Cin1 * Crn1 * s)

    M_all[:, 1, 2] = -C22 * (Crn1 * s - Cin1 * c + Cin1**2 * s -
                             Crn1**2 * s + 2 * Cin1 * Crn1 * c)

    M_all[:, 1, 3] = C22 * (Crn1 * c + Cin1 * s + Cin1**2 * c -
                            Crn1**2 * c - 2 * Cin1 * Crn1 * s)

    M_all[:, 1, 4] = 2 * C11 * C22 * (Cin1 * Crn1 * c**2 -
                                      Cin1 * Crn1 * s**2 +
                                      Cin1**2 * c * s -
                                      Crn1**2 * c * s)

    M_all[:, 1, 5] = M_all[:, 0, 4]
    M_all[:, 1, 6] = M_all[:, 0, 4]

    M_all[:, 1, 7] = -2 * C11 * C22 * (Cin1 * Crn1 * c**2 -
                                       Cin1 * Crn1 * s**2 +
                                       Cin1**2 * c * s -
                                       Crn1**2 * c * s)

    M_all[:, 1, 8] = -2 * C12 * C21 * (Cin1 * Crn1 * c**2 -
                                       Cin1 * Crn1 * s**2 +
                                       Cin1**2 * c * s -
                                       Crn1**2 * c * s)

    M_all[:, 1, 9] = -C12 * C21 * (Crn1**2 * c**2 - Cin1**2 * c**2 +
                                   Cin1**2 * s**2 - Crn1**2 * s**2 +
                                   4 * Cin1 * Crn1 * c * s)

    M_all[:, 1, 10] = M_all[:, 1, 9]

    M_all[:, 1, 11] = 2 * C12 * C21 * (Cin1 * Crn1 * c**2 -
                                       Cin1 * Crn1 * s**2 +
                                       Cin1**2 * c * s -
                                       Crn1**2 * c * s)

    # ---- b_all (2 components per i) ----
    b_all[:, 0] = -(Cr2n1 - Crn1 - Cin1**2 + Crn1**2 +
                    Ci2n1 * Cin1 - Cr2n1 * Crn1)

    b_all[:, 1] = -(Ci2n1 - Cin1 - Ci2n1 * Crn1 -
                    Cin1 * Cr2n1 + 2 * Cin1 * Crn1)

    return M_all, b_all


def build_mu_from_p(p: np.ndarray, enforce_symmetry: bool) -> np.ndarray:
    """
    Map parameter vector p (6 or 8 real parameters) to the 12-component μ vector
    used in the M μ - b system.
    """

    if enforce_symmetry:
        mur11, mui11, mur22, mui22, mur12, mui12 = p
        mur21, mui21 = mur12, mui12
    else:
        mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21 = p

    mu = np.array([
        mur11, mui11, mur22, mui22,
        mur11 * mur22, mur11 * mui22, mui11 * mur22, mui11 * mui22,
        mur12 * mur21, mur12 * mui21, mui12 * mur21, mui12 * mui21
    ])

    return mu

def unpack_p_to_values(p: np.ndarray, enforce_symmetry: bool):
    """
    Unpack p (6 or 8 real parameters) into the 8 underlying μ_ij components:
    (mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21)
    """

    if enforce_symmetry:
        mur11, mui11, mur22, mui22, mur12, mui12 = p
        mur21, mui21 = mur12, mui12
    else:
        mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21 = p

    return mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21

def find_mu_from_block(M_block: np.ndarray,
                       b_block: np.ndarray,
                       enforce_symmetry: bool,
                       quiet: bool = False):
    """
    Nonlinear least-squares solver for a single sample, using a precomputed
    M_block (2x12) and b_block (2,).

    Returns:
        mu_vec : (12,)  nonlinear μ combinations
        mvals  : (8,)   underlying (mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21)
    """

    # Initial guess for p
    if enforce_symmetry:
        p0 = np.random.random_sample(6)  # [mur11, mui11, mur22, mui22, mur12, mui12]
        args_enforce = True
    else:
        p0 = np.random.random_sample(8)  # [mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21]
        args_enforce = False

    def residual_local(p):
        mu = build_mu_from_p(p, args_enforce)   # (12,)
        return M_block @ mu - b_block           # (2,)

    res = least_squares(residual_local, p0, method="trf")

    if not res.success and not quiet:
        print("find_mu_from_block: success:", res.success,
              "status:", res.status, "cost:", res.cost)
        print("x (result):", res.x)

    p_opt = res.x
    mu_vec = build_mu_from_p(p_opt, args_enforce)
    mvals = np.array(unpack_p_to_values(p_opt, args_enforce), dtype=float)

    return mu_vec, mvals
