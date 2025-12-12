import numpy as np
from scipy.optimize import least_squares
from utils import logger

# ============================================================
#          SECOND-ORDER μ-FIT  —  CLEAN PER-R VERSION
# ============================================================
def cond_safe(A):
    try:
        return float(np.linalg.cond(A))
    except Exception:
        return float(np.inf)

def scale_columns_like_global(A, rel_thresh):
    """
    Per-block version of the same scaling approach currently used globally:
      effective_norms = max(col_norm, rel_thresh * max_norm)
      A_scaled = A / effective_norms
    Returns (A_scaled, effective_norms).
    """
    col_norms = np.linalg.norm(A, axis=0)
    max_norm = np.max(col_norms) if np.any(col_norms > 0) else 1.0
    min_allowed = rel_thresh * max_norm
    effective_norms = np.maximum(col_norms, min_allowed)
    A_scaled = A / effective_norms
    return A_scaled, effective_norms

def svd_rank(A, rel_thresh=1e-10):
    """
    Numerical rank based on singular values >= rel_thresh * sigma_max.
    Returns (rank, sigma_max, sigma_min, ratio, svals).
    """
    try:
        svals = np.linalg.svd(A, compute_uv=False)
        if svals.size == 0 or svals[0] <= 0:
            return 0, np.nan, np.nan, np.nan, svals
        sigma_max = float(svals[0])
        sigma_min = float(svals[-1])
        rank = int(np.sum(svals >= rel_thresh * sigma_max))
        ratio = float(sigma_min / sigma_max) if sigma_max > 0 else np.nan
        return rank, sigma_max, sigma_min, ratio, svals
    except Exception:
        return 0, np.nan, np.nan, np.nan, np.array([])


# ------------------------------------------------------------
#  Stable normalization using complex ratios s² / s_ref²
# ------------------------------------------------------------
def compute_normalized_terms(sigma, w, s_ref, s_other):
    """
    Compute normalized Cr/Ci-like quantities using the stable
    complex-ratio representation:

        (sigma + i w)^2 / (s_ref)^2

    This eliminates the catastrophic cancellation problems found
    in the classical Cr/Ci formulation.
    """

    sigma = np.asarray(sigma, float)
    w = np.asarray(w, float)

    s = sigma + 1j * w
    s_ref_c = s_ref.real + 1j * s_ref.imag
    s_other_c = s_other.real + 1j * s_other.imag

    s2 = s * s
    s_ref2 = s_ref_c * s_ref_c
    s_other2 = s_other_c * s_other_c

    eps = 1e-12
    if abs(s_ref2) < eps:
        logger.warning(
            f"[compute_normalized_terms] s_ref^2 extremely small: "
            f"s_ref={s_ref}, s_ref^2={s_ref2}"
        )
        s_ref2 = eps

    ratio = s2 / s_ref2
    ratio_other = s_other2 / s_ref2

    Crn = ratio.real
    Cin = ratio.imag
    Cr_other = ratio_other.real
    Cin_other = ratio_other.imag

    return Crn, Cin, Cr_other, Cin_other


# ------------------------------------------------------------
#    Vectorized M and b construction for all n samples
# ------------------------------------------------------------
def create_M11M22minusM12M21_vectorized(config, tau, w, sigma, s_1, s_2, n):
    """
    Build M(i) and b(i) for all n samples at once.
    
    Returns
    -------
    M_all : (m, 2, 12)
    b_all : (m, 2)
    """
    m = len(n) - 1
    idx = np.arange(m)

    alphaK1 = config.alpha[0] * config.K
    alphaK2 = config.alpha[1] * config.K

    exp_term = np.exp(-tau * sigma[idx])

    C11 = config.Lambda[0] * alphaK1 * config.nu[0] * n[1:] * exp_term
    C22 = config.Lambda[1] * alphaK2 * config.nu[1] * n[1:] * exp_term
    C12 = config.Lambda[1] * alphaK1 * config.nu[2] * n[1:] * exp_term
    C21 = config.Lambda[0] * alphaK2 * config.nu[3] * n[1:] * exp_term

    # -- Stable normalization
    Crn, Cin, Cr2n, Ci2n = compute_normalized_terms(
        sigma[idx], w[idx], s_1, s_2
    )

    s = np.sin(w[idx] * tau)
    c = np.cos(w[idx] * tau)

    M_all = np.zeros((m, 2, 12), float)
    b_all = np.zeros((m, 2), float)

    # --------------- Construct M for all rows ---------------
    # (Identical formulas as original, just vectorized)
    M_all[:, 0, 0] = C11*(Cin**2*c - Crn**2*c - Ci2n*Cin*c + Cr2n*Crn*c +
                         Ci2n*Crn*s + Cin*Cr2n*s - 2*Cin*Crn*s)

    M_all[:, 0, 1] = -C11*(Crn**2*s - Cin**2*s + Ci2n*Crn*c + Cin*Cr2n*c -
                           2*Cin*Crn*c + Ci2n*Cin*s - Cr2n*Crn*s)

    M_all[:, 0, 2] = C22*(Crn*c + Cin*s + Cin**2*c -
                          Crn**2*c - 2*Cin*Crn*s)

    M_all[:, 0, 3] = C22*(Crn*s - Cin*c + Cin**2*s -
                          Crn**2*s + 2*Cin*Crn*c)

    M_all[:, 0, 4] = C11*C22*(Crn**2*c**2 - Cin**2*c**2 +
                              Cin**2*s**2 - Crn**2*s**2 +
                              4*Cin*Crn*c*s)

    M_all[:, 0, 5] = -2*C11*C22*(Cin*Crn*c**2 - Cin*Crn*s**2 +
                                 Cin**2*c*s - Crn**2*c*s)

    M_all[:, 0, 6] = M_all[:, 0, 5]
    M_all[:, 0, 7] = -M_all[:, 0, 4]

    M_all[:, 0, 8] = -C12*C21*(Crn**2*c**2 - Cin**2*c**2 +
                               Cin**2*s**2 - Crn**2*s**2 +
                               4*Cin*Crn*c*s)

    M_all[:, 0, 9] = 2*C12*C21*(Cin*Crn*c**2 - Cin*Crn*s**2 +
                                Cin**2*c*s - Crn**2*c*s)

    M_all[:, 0, 10] = M_all[:, 0, 9]
    M_all[:, 0, 11] = -M_all[:, 0, 8]

    # --- second row ---
    M_all[:, 1, 0] = C11*(Crn**2*s - Cin**2*s + Ci2n*Crn*c +
                         Cin*Cr2n*c - 2*Cin*Crn*c +
                         Ci2n*Cin*s - Cr2n*Crn*s)

    M_all[:, 1, 1] = C11*(Cin**2*c - Crn**2*c - Ci2n*Cin*c +
                         Cr2n*Crn*c + Ci2n*Crn*s +
                         Cin*Cr2n*s - 2*Cin*Crn*s)

    M_all[:, 1, 2] = -C22*(Crn*s - Cin*c + Cin**2*s -
                           Crn**2*s + 2*Cin*Crn*c)

    M_all[:, 1, 3] = C22*(Crn*c + Cin*s + Cin**2*c -
                          Crn**2*c - 2*Cin*Crn*s)

    M_all[:, 1, 4] = 2*C11*C22*(Cin*Crn*c**2 - Cin*Crn*s**2 +
                                Cin**2*c*s - Crn**2*c*s)

    M_all[:, 1, 5] = M_all[:, 0, 4]
    M_all[:, 1, 6] = M_all[:, 0, 4]

    M_all[:, 1, 7] = -2*C11*C22*(Cin*Crn*c**2 - Cin*Crn*s**2 +
                                 Cin**2*c*s - Crn**2*c*s)

    M_all[:, 1, 8] = -2*C12*C21*(Cin*Crn*c**2 - Cin*Crn*s**2 +
                                 Cin**2*c*s - Crn**2*c*s)

    M_all[:, 1, 9] = -C12*C21*(Crn**2*c**2 - Cin**2*c**2 +
                               Cin**2*s**2 - Crn**2*s**2 +
                               4*Cin*Crn*c*s)

    M_all[:, 1, 10] = M_all[:, 1, 9]
    M_all[:, 1, 11] = 2*C12*C21*(Cin*Crn*c**2 - Cin*Crn*s**2 +
                                 Cin**2*c*s - Crn**2*c*s)

    # --- b(i) ---
    b_all[:, 0] = -(Cr2n - Crn - Cin**2 + Crn**2 +
                    Ci2n*Cin - Cr2n*Crn)

    b_all[:, 1] = -(Ci2n - Cin - Ci2n*Crn -
                    Cin*Cr2n + 2*Cin*Crn)

    return M_all, b_all


# ------------------------------------------------------------
#  Map p → symbolic μ-vector (12 components)
# ------------------------------------------------------------
# ------------------------------------------------------------
#  Mapping between optimization variables q and physical p, μ
# ------------------------------------------------------------

def q_to_p(q, enforce_symmetry):
    if enforce_symmetry:
        a11, phi11, a22, phi22, a12, phi12 = q
        a21, phi21 = a12, phi12
    else:
        a11, phi11, a22, phi22, a12, phi12, a21, phi21 = q

    mu11 = np.exp(a11) * (np.cos(phi11) + 1j * np.sin(phi11))
    mu22 = np.exp(a22) * (np.cos(phi22) + 1j * np.sin(phi22))
    mu12 = np.exp(a12) * (np.cos(phi12) + 1j * np.sin(phi12))
    mu21 = np.exp(a21) * (np.cos(phi21) + 1j * np.sin(phi21))

    if enforce_symmetry:
        return np.array([
            mu11.real, mu11.imag,
            mu22.real, mu22.imag,
            mu12.real, mu12.imag,
        ])
    else:
        return np.array([
            mu11.real, mu11.imag,
            mu22.real, mu22.imag,
            mu12.real, mu12.imag,
            mu21.real, mu21.imag,
        ])

def build_mu_from_p_phys(p, enforce_symmetry):
    if enforce_symmetry:
        mur11, mui11, mur22, mui22, mur12, mui12 = p
        mur21, mui21 = mur12, mui12
    else:
        mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21 = p

    return np.array([
        mur11, mui11, mur22, mui22,
        mur11*mur22, mur11*mui22, mui11*mur22, mui11*mui22,
        mur12*mur21, mur12*mui21, mui12*mur21, mui12*mui21,
    ])


def evaluate_polynomial_p(R, theta, degree, enforce_symmetry):
    """
    Evaluate p(R) using polynomial coefficients theta.
    theta is structured as (p_dim, degree+1).
    """
    theta = np.asarray(theta)
    if enforce_symmetry:
        p_dim = 6
    else:
        p_dim = 8

    coeffs = theta.reshape(p_dim, degree+1)  # (6,2) or (8,2)
    powers = np.array([R**j for j in range(degree+1)])  # [1, R]
    return coeffs @ powers




# ============================================================
#   MAIN GLOBAL PER-R SOLVER  —  the only nonlinear solve
# ============================================================
def mu_array_stacked(
    config,
    n,
    tau,
    w_big,
    sigma_big,
    s_1,
    s_2,
    use_only_acoustic,
    enforce_symmetry,
    weights=None,
    quiet=False,
    extra_branches=None,
):
    mode = getattr(config, "mu_solver_mode", "two_stage")
    two_stage = (mode == "two_stage")


    # --------------------------------------------------
    # helpers
    # --------------------------------------------------
    def ensure_2d(arr):
        arr = np.asarray(arr)
        return arr[:, None] if arr.ndim == 1 else arr

    def complex_to_q(mu):
        r = max(abs(mu), 1e-12)
        return np.log(r), np.arctan2(mu.imag, mu.real)

    def qdiag_to_p_fixed(qd):
        a11, phi11, a22, phi22 = qd
        mu11 = np.exp(a11) * (np.cos(phi11) + 1j*np.sin(phi11))
        mu22 = np.exp(a22) * (np.cos(phi22) + 1j*np.sin(phi22))
        mu12 = mu12_prior
        mu21 = mu12_prior if enforce_symmetry else mu21_prior

        if enforce_symmetry:
            return np.array([
                mu11.real, mu11.imag,
                mu22.real, mu22.imag,
                mu12.real, mu12.imag,
            ])
        else:
            return np.array([
                mu11.real, mu11.imag,
                mu22.real, mu22.imag,
                mu12.real, mu12.imag,
                mu21.real, mu21.imag,
            ])

    # --------------------------------------------------
    # config
    # --------------------------------------------------
    two_stage = getattr(config, "mu_two_stage")
    mu12_prior = getattr(config, "mu12_prior", 1.0 + 0.0j)
    mu21_prior = getattr(config, "mu21_prior", mu12_prior)

    rel_col_thresh = getattr(config, "mu_col_rel_thresh", 1e-3)
    rel_svd_thresh = getattr(config, "mu_svd_rel_thresh", 1e-10)

    # --------------------------------------------------
    # shapes
    # --------------------------------------------------
    w_big = ensure_2d(w_big)
    sigma_big = ensure_2d(sigma_big)

    if use_only_acoustic:
        w_big = w_big[:, :1]
        sigma_big = sigma_big[:, :1]

    m, T = w_big.shape

    # --------------------------------------------------
    # build M, b + diagnostics
    # --------------------------------------------------
    M_list, b_list = [], []
    block_info = []

    for j in range(T):
        # -------- branch 1 --------
        Mj, bj = create_M11M22minusM12M21_vectorized(
            config, tau, w_big[:, j], sigma_big[:, j], s_1, s_2, n
        )
        M_list.append(Mj)
        b_list.append(bj)

        M2d = Mj.reshape(-1, 12)
        M2d_s, _ = scale_columns_like_global(M2d, rel_col_thresh)

        r_sc, smax, smin, ratio, _ = svd_rank(M2d_s, rel_thresh=rel_svd_thresh)

        block_info.append({
            "block": "branch1",
            "time_idx": j,
            "cond_scaled": cond_safe(M2d_s),
            "rank_scaled": r_sc,
            "svd_sigma_min_scaled": smin,
            "svd_ratio_scaled": ratio,
        })

        # -------- branch 2+ --------
        if extra_branches is not None:
            for b_idx, br in enumerate(extra_branches, start=2):
                w_ex = ensure_2d(br["w_big"])
                sigma_ex = ensure_2d(br["sigma_big"])
                s_ref = br.get("s_ref", s_2)
                s_other = br.get("s_other", s_1)

                Mex, bex = create_M11M22minusM12M21_vectorized(
                    config, tau, w_ex[:, j], sigma_ex[:, j], s_ref, s_other, n
                )

                M_list.append(Mex)
                b_list.append(bex)

                M2d_ex = Mex.reshape(-1, 12)
                M2d_ex_s, _ = scale_columns_like_global(M2d_ex, rel_col_thresh)

                r_sc, smax, smin, ratio, _ = svd_rank(
                    M2d_ex_s, rel_thresh=rel_svd_thresh
                )

                block_info.append({
                    "block": f"branch{b_idx}",
                    "time_idx": j,
                    "cond_scaled": cond_safe(M2d_ex_s),
                    "rank_scaled": r_sc,
                    "svd_sigma_min_scaled": smin,
                    "svd_ratio_scaled": ratio,
                })

    # --------------------------------------------------
    # stack
    # --------------------------------------------------
    M_all = np.concatenate(M_list, axis=0)
    b_all = np.concatenate(b_list, axis=0)

    M_2d = M_all.reshape(-1, 12)
    b_2d = b_all.reshape(-1)

    col_norms = np.linalg.norm(M_2d, axis=0)
    max_norm = np.max(col_norms) if np.any(col_norms > 0) else 1.0
    eff_norms = np.maximum(col_norms, rel_col_thresh * max_norm)

    M_scaled = M_2d / eff_norms
    S12 = eff_norms

    cond_M = cond_safe(M_2d)
    cond_M_scaled = cond_safe(M_scaled)

    rank_svd, smax, smin, ratio, svals = svd_rank(
        M_scaled, rel_thresh=rel_svd_thresh
    )

    # --------------------------------------------------
    # residuals
    # --------------------------------------------------
    def residual_global(q):
        p = q_to_p(q, enforce_symmetry)
        mu_vec = build_mu_from_p_phys(p, enforce_symmetry)
        mu_eff = S12 * mu_vec
        Rv = M_scaled @ mu_eff - b_2d

        p_target = (
            np.array([1,0,1,0,1,0])
            if enforce_symmetry
            else np.array([1,0,1,0,1,0,1,0])
        )

        lam = getattr(config, "mu_reg_lambda", 0.05)
        reg = np.sqrt(lam) * (p - p_target)
        return np.concatenate([Rv, reg])

    # --------------------------------------------------
    # initial guess + bounds
    # --------------------------------------------------
    q_dim = 6 if enforce_symmetry else 8
    q0 = np.zeros(q_dim)

    a12, phi12 = complex_to_q(mu12_prior)
    q0[4:6] = [a12, phi12]
    if not enforce_symmetry:
        a21, phi21 = complex_to_q(mu21_prior)
        q0[6:8] = [a21, phi21]

    a_min = getattr(config, "mu_logmag_min", -0.7)
    a_max = getattr(config, "mu_logmag_max",  0.7)
    phi_max = getattr(config, "mu_phase_max", np.pi)

    a_min_c = getattr(config, "mu_cpl_logmag_min", a_min)
    a_max_c = getattr(config, "mu_cpl_logmag_max", a_max)
    phi_max_c = getattr(config, "mu_cpl_phase_max", phi_max)

    lower = np.zeros(q_dim)
    upper = np.zeros(q_dim)

    lower[:4:2] = a_min
    upper[:4:2] = a_max
    lower[1:4:2] = -phi_max
    upper[1:4:2] =  phi_max

    lower[4::2] = a_min_c
    upper[4::2] = a_max_c
    lower[5::2] = -phi_max_c
    upper[5::2] =  phi_max_c

    # --------------------------------------------------
    # STAGE 1 (diag only)
    # --------------------------------------------------
    # ---------------------------
    # Solve
    # ---------------------------
    if two_stage:
        # ---- Stage 1: diagonals only ----
        def residual_stage1(qd):
            p = qdiag_to_p_fixed(qd)
            mu_vec = build_mu_from_p_phys(p, enforce_symmetry)
            mu_eff = S12 * mu_vec
            Rv = M_scaled @ mu_eff - b_2d

            lam = getattr(config, "mu_reg_lambda_diag", 0.05)
            reg = np.sqrt(lam) * (p[:4] - np.array([1,0,1,0]))
            return np.concatenate([Rv, reg])

        res1 = least_squares(
            residual_stage1,
            q0[:4],
            method="trf",
            bounds=(lower[:4], upper[:4]),
        )
        q0[:4] = res1.x

    # ---- Stage 2 (full) : always executed ----
    res = least_squares(
        residual_global,
        q0,
        method="trf",
        bounds=(lower, upper),
    )

    q_opt = res.x
    p_opt = q_to_p(q_opt, enforce_symmetry)



    # --------------------------------------------------
    # summaries
    # --------------------------------------------------
    def summarize(branch):
        vals = [d for d in block_info if d["block"] == branch]
        if not vals:
            return None
        return {
            "rank_scaled_min": int(np.min([v["rank_scaled"] for v in vals])),
            "rank_scaled_max": int(np.max([v["rank_scaled"] for v in vals])),
            "svd_ratio_scaled_min": float(np.min([v["svd_ratio_scaled"] for v in vals])),
            "cond_scaled_min": float(np.min([v["cond_scaled"] for v in vals])),
            "cond_scaled_max": float(np.max([v["cond_scaled"] for v in vals])),
            "num_blocks": len(vals),
            "num_with_rank": len(vals),
        }

    info = dict(
        cond_M=cond_M,
        cond_M_scaled=cond_M_scaled,
        rank_svd=rank_svd,
        svd_sigma_max=smax,
        svd_sigma_min=smin,
        svd_ratio=ratio,
        svd_svals=svals,
        mu_svd_rel_thresh=rel_svd_thresh,
        global_cost=res.cost,
        residual_global_norm=np.linalg.norm(residual_global(q_opt)),
        q_opt=q_opt,
        two_stage=two_stage,
        block_info=block_info,
        branch1_summary=summarize("branch1"),
        branch2_summary=summarize("branch2"),
    )

    mu_stacked = np.full((m*T, 12), np.nan)
    return mu_stacked, p_opt, info

