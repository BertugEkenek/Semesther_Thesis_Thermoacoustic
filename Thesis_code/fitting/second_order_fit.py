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
    col_norms = np.linalg.norm(A, axis=0)
    max_norm = np.max(col_norms) if np.any(col_norms > 0) else 1.0
    min_allowed = rel_thresh * max_norm
    effective_norms = np.maximum(col_norms, min_allowed)
    A_scaled = A / effective_norms
    return A_scaled, effective_norms

def svd_rank(A, rel_thresh=1e-10):
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
def compute_normalized_terms(sigma, w, s_ref, s_1, s_2):
    sigma = np.asarray(sigma, float)
    w = np.asarray(w, float)

    s = sigma + 1j * w

    s_ref2 = (s_ref.real + 1j*s_ref.imag)**2
    s1_2   = (s_1.real  + 1j*s_1.imag )**2
    s2_2   = (s_2.real  + 1j*s_2.imag )**2

    eps = 1e-12
    if abs(s_ref2) < eps:
        logger.warning(f"[compute_normalized_terms] s_ref^2 too small: s_ref={s_ref}")
        s_ref2 = eps

    ratio_s = (s*s) / s_ref2
    ratio_1 = s1_2 / s_ref2
    ratio_2 = s2_2 / s_ref2

    return (
        ratio_s.real, ratio_s.imag,
        ratio_1.real, ratio_1.imag,
        ratio_2.real, ratio_2.imag,
    )


# ------------------------------------------------------------
#    Vectorized M and b construction for all n samples
# ------------------------------------------------------------
def create_M11M22minusM12M21_vectorized(config, tau, w, sigma, s_ref, s_1, s_2, n):
    m = len(n) - 1
    idx = np.arange(m)

    alphaK1 = config.alpha[0] * config.K
    alphaK2 = config.alpha[1] * config.K

    exp_term = np.exp(-tau * sigma[idx])

    C11 = config.Lambda[0] * alphaK1 * config.nu[0] * n[1:] * exp_term
    C22 = config.Lambda[1] * alphaK2 * config.nu[1] * n[1:] * exp_term
    C12 = config.Lambda[1] * alphaK2 * config.nu[2] * n[1:] * exp_term
    C21 = config.Lambda[0] * alphaK1 * config.nu[3] * n[1:] * exp_term

    # NEW: symmetric normalized terms
    Crn, Cin, Cr1, Ci1, Cr2, Ci2 = compute_normalized_terms(
        sigma[idx], w[idx], s_ref, s_1, s_2
    )

    s = np.sin(w[idx] * tau)
    c = np.cos(w[idx] * tau)

    M_all = np.zeros((m, 2, 12), float)
    b_all = np.zeros((m, 2), float)

    # --- first row ---
    # μ11 terms: depend on (Cr2, Ci2) (other diagonal constant)
    M_all[:, 0, 0] = C11*(Cin**2*c - Crn**2*c - Ci2*Cin*c + Cr2*Crn*c +
                         Ci2*Crn*s + Cin*Cr2*s - 2*Cin*Crn*s)

    M_all[:, 0, 1] = -C11*(Crn**2*s - Cin**2*s + Ci2*Crn*c + Cin*Cr2*c -
                           2*Cin*Crn*c + Ci2*Cin*s - Cr2*Crn*s)

    # μ22 terms: symmetric counterpart, depend on (Cr1, Ci1)
    M_all[:, 0, 2] = C22*(Cin**2*c - Crn**2*c - Ci1*Cin*c + Cr1*Crn*c +
                          Ci1*Crn*s + Cin*Cr1*s - 2*Cin*Crn*s)

    M_all[:, 0, 3] = -C22*(Crn**2*s - Cin**2*s + Ci1*Crn*c + Cin*Cr1*c -
                           2*Cin*Crn*c + Ci1*Cin*s - Cr1*Crn*s)

    # bilinear diagonal product (unchanged)
    M_all[:, 0, 4] = C11*C22*(Crn**2*c**2 - Cin**2*c**2 +
                              Cin**2*s**2 - Crn**2*s**2 +
                              4*Cin*Crn*c*s)

    M_all[:, 0, 5] = -2*C11*C22*(Cin*Crn*c**2 - Cin*Crn*s**2 +
                                 Cin**2*c*s - Crn**2*c*s)

    M_all[:, 0, 6] = M_all[:, 0, 5]
    M_all[:, 0, 7] = -M_all[:, 0, 4]

    # coupling product (unchanged)
    M_all[:, 0, 8] = -C12*C21*(Crn**2*c**2 - Cin**2*c**2 +
                               Cin**2*s**2 - Crn**2*s**2 +
                               4*Cin*Crn*c*s)

    M_all[:, 0, 9] = 2*C12*C21*(Cin*Crn*c**2 - Cin*Crn*s**2 +
                                Cin**2*c*s - Crn**2*c*s)

    M_all[:, 0, 10] = M_all[:, 0, 9]
    M_all[:, 0, 11] = -M_all[:, 0, 8]

    # --- second row ---
    M_all[:, 1, 0] = C11*(Crn**2*s - Cin**2*s + Ci2*Crn*c +
                         Cin*Cr2*c - 2*Cin*Crn*c +
                         Ci2*Cin*s - Cr2*Crn*s)

    M_all[:, 1, 1] = C11*(Cin**2*c - Crn**2*c - Ci2*Cin*c +
                         Cr2*Crn*c + Ci2*Crn*s +
                         Cin*Cr2*s - 2*Cin*Crn*s)

    M_all[:, 1, 2] = -C22*(Crn**2*s - Cin**2*s + Ci1*Crn*c +
                           Cin*Cr1*c - 2*Cin*Crn*c +
                           Ci1*Cin*s - Cr1*Crn*s)

    M_all[:, 1, 3] = C22*(Cin**2*c - Crn**2*c - Ci1*Cin*c +
                          Cr1*Crn*c + Ci1*Crn*s +
                          Cin*Cr1*s - 2*Cin*Crn*s)

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

    # --- b(i): NEW symmetric constant term (move to RHS => negate)
    # Constant in Re(det):  -Ci1*Ci2 + Ci1*Cin + Ci2*Cin - Cin^2 + Cr1*Cr2 - Cr1*Crn - Cr2*Crn + Crn^2
    b_all[:, 0] = -(
        -Ci1*Ci2 + Ci1*Cin + Ci2*Cin - Cin**2
        + Cr1*Cr2 - Cr1*Crn - Cr2*Crn + Crn**2
    )

    # Constant in Im(det):  Ci1*Cr2 - Ci1*Crn + Ci2*Cr1 - Ci2*Crn - Cin*Cr1 - Cin*Cr2 + 2*Cin*Crn
    b_all[:, 1] = -(
        Ci1*Cr2 - Ci1*Crn + Ci2*Cr1 - Ci2*Crn
        - Cin*Cr1 - Cin*Cr2 + 2*Cin*Crn
    )

    return M_all, b_all


# ------------------------------------------------------------
#  Mapping between optimization variables q and physical p
# ------------------------------------------------------------
def q_to_p(q, enforce_symmetry, hard_constraint=False, sign=+1):
    mu11, mu22, mu12, mu21 = q_to_mu_complex(
        q, enforce_symmetry, hard_constraint, sign
    )

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

def q_to_mu_complex(q, enforce_symmetry, hard_constraint=False, sign=+1):
    if hard_constraint:
        a11, phi11, a22, phi22 = q
        mu11 = np.exp(a11) * (np.cos(phi11) + 1j*np.sin(phi11))
        mu22 = np.exp(a22) * (np.cos(phi22) + 1j*np.sin(phi22))

        # branch-stable coupling (avoid complex sqrt branch cuts)
        a12 = 0.5 * (a11 + a22)
        phi12 = 0.5 * (phi11 + phi22)
        mu12_base = np.exp(a12) * (np.cos(phi12) + 1j*np.sin(phi12))

        mu12 = sign * mu12_base
        mu21 = mu12
        return mu11, mu22, mu12, mu21


    # existing behavior
    if enforce_symmetry:
        a11, phi11, a22, phi22, a12, phi12 = q
        a21, phi21 = a12, phi12
    else:
        a11, phi11, a22, phi22, a12, phi12, a21, phi21 = q

    mu11 = np.exp(a11) * (np.cos(phi11) + 1j*np.sin(phi11))
    mu22 = np.exp(a22) * (np.cos(phi22) + 1j*np.sin(phi22))
    mu12 = np.exp(a12) * (np.cos(phi12) + 1j*np.sin(phi12))
    mu21 = np.exp(a21) * (np.cos(phi21) + 1j*np.sin(phi21))
    return mu11, mu22, mu12, mu21



# ============================================================
#   MAIN GLOBAL PER-R SOLVER (RENAMED)
# ============================================================
def solve_mu_per_R(
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
    init_mu11: complex | None = None,
    init_mu22: complex | None = None,
):
    hard = bool(getattr(config, "mu_hard_constraint", False))

    # =========================================================
    # helpers
    # =========================================================
    def ensure_2d(arr):
        arr = np.asarray(arr)
        return arr[:, None] if arr.ndim == 1 else arr

    def complex_to_q(mu):
        r = max(abs(mu), 1e-12)
        return np.log(r), np.arctan2(mu.imag, mu.real)

    def get_time_weight(j, T_local):
        if weights is None:
            return 1.0
        wv = np.asarray(weights).ravel()
        if wv.size == 1:
            return float(wv[0])
        if wv.size != T_local:
            raise ValueError(
                f"weights must have length T={T_local} (or be scalar). Got {wv.size}."
            )
        return float(wv[j])
    
    def build_mu_weight_vector(enforce_symmetry):
        """
        Returns per-parameter weights for (p - p_target).
        Order must match p-vector layout.
        """
        wcfg = getattr(config, "mu_target_weights", {})

        w11 = float(wcfg.get("mu11", 1.0))
        w22 = float(wcfg.get("mu22", 1.0))
        w12 = float(wcfg.get("mu12", 1.0))
        w21 = float(wcfg.get("mu21", w12))

        if enforce_symmetry:
            # [Re11, Im11, Re22, Im22, Re12, Im12]
            return np.array([
                w11, w11,
                w22, w22,
                w12, w12,
            ], dtype=float)
        else:
            # [Re11, Im11, Re22, Im22, Re12, Im12, Re21, Im21]
            return np.array([
                w11, w11,
                w22, w22,
                w12, w12,
                w21, w21,
            ], dtype=float)
        
    def complex_to_q(mu):
        r = max(abs(mu), 1e-12)
        return np.log(r), np.arctan2(mu.imag, mu.real)



    # =========================================================
    # config
    # =========================================================
    mu12_prior = getattr(config, "mu12_prior", 1.0 + 0.0j)
    mu21_prior = getattr(config, "mu21_prior", mu12_prior)

    rel_col_thresh = getattr(config, "mu_col_rel_thresh", 1e-3)
    rel_svd_thresh = getattr(config, "mu_svd_rel_thresh", 1e-10)

    # =========================================================
    # shapes (ALWAYS normalize to 2-D: (m, T))
    # =========================================================
    w_big = ensure_2d(w_big)
    sigma_big = ensure_2d(sigma_big)

    if use_only_acoustic:
        # acoustic-only means keep only the first "time column"
        w_big = w_big[:, :1]
        sigma_big = sigma_big[:, :1]

    m, T = w_big.shape

    # Normalize extra branches to same (m, T) convention
    norm_extra = None
    if extra_branches is not None:
        norm_extra = []
        for br in extra_branches:
            w_ex = ensure_2d(br["w_big"])
            s_ex = ensure_2d(br["sigma_big"])

            if use_only_acoustic:
                w_ex = w_ex[:, :1]
                s_ex = s_ex[:, :1]

            if w_ex.shape[1] != T or s_ex.shape[1] != T:
                if T == 1:
                    w_ex = w_ex[:, :1]
                    s_ex = s_ex[:, :1]
                else:
                    raise ValueError(
                        f"Extra branch shape mismatch: main T={T}, "
                        f"extra w_big shape={w_ex.shape}, sigma_big shape={s_ex.shape}"
                    )

            norm_extra.append(
                dict(
                    w_big=w_ex,
                    sigma_big=s_ex,
                    s_ref=br.get("s_ref", s_2),
                    s_other=br.get("s_other", s_1),
                )
            )

    # =========================================================
    # build M, b (stack over time and over branches)
    # apply weights as row scaling (Fix 2)
    # =========================================================
    block_info = []
    M_list, b_list = [], []

    for j in range(T):
        wj = get_time_weight(j, T)

        # ---------- branch 1 ----------
        Mj, bj = create_M11M22minusM12M21_vectorized(
            config, tau, w_big[:, j], sigma_big[:, j],
            s_ref=s_1, s_1=s_1, s_2=s_2, n=n
        )


        # apply time-weight
        if wj != 1.0:
            Mj = wj * Mj
            bj = wj * bj

        M_list.append(Mj)
        b_list.append(bj)

        # diagnostics only if not quiet (Fix 3)
        if not quiet:
            M2d = Mj.reshape(-1, 12)
            M2d_s, _ = scale_columns_like_global(M2d, rel_col_thresh)
            r_sc, _, smin, ratio, _ = svd_rank(M2d_s, rel_thresh=rel_svd_thresh)

            block_info.append(dict(
                block="branch1",
                time_idx=j,
                cond_scaled=cond_safe(M2d_s),
                rank_scaled=r_sc,
                svd_sigma_min_scaled=smin,
                svd_ratio_scaled=ratio,
            ))

        # ---------- extra branches ----------
        if norm_extra is not None:
            for b_idx, br in enumerate(norm_extra, start=2):
                w_ex = br["w_big"]
                s_ex = br["sigma_big"]
                Mex, bex = create_M11M22minusM12M21_vectorized(
                    config, tau, w_ex[:, j], s_ex[:, j],
                    s_ref=br["s_ref"], s_1=s_1, s_2=s_2, n=n
                )

                if wj != 1.0:
                    Mex = wj * Mex
                    bex = wj * bex

                M_list.append(Mex)
                b_list.append(bex)

                if not quiet:
                    M2d_ex = Mex.reshape(-1, 12)
                    M2d_ex_s, _ = scale_columns_like_global(M2d_ex, rel_col_thresh)
                    r_sc, _, smin, ratio, _ = svd_rank(M2d_ex_s, rel_thresh=rel_svd_thresh)

                    block_info.append(dict(
                        block=f"branch{b_idx}",
                        time_idx=j,
                        cond_scaled=cond_safe(M2d_ex_s),
                        rank_scaled=r_sc,
                        svd_sigma_min_scaled=smin,
                        svd_ratio_scaled=ratio,
                    ))

    M_all = np.concatenate(M_list, axis=0)
    b_all = np.concatenate(b_list, axis=0)

    M_2d = M_all.reshape(-1, 12)
    b_2d = b_all.reshape(-1)

    col_norms = np.linalg.norm(M_2d, axis=0)
    max_norm = np.max(col_norms) if np.any(col_norms > 0) else 1.0
    eff_norms = np.maximum(col_norms, rel_col_thresh * max_norm)

    M_scaled = M_2d / eff_norms
    S12 = eff_norms  # scale vector (12,)

    # =========================================================
    # internal solver for one sign (hard) or nominal (+) (soft)
    # =========================================================
    def solve_for_sign(sign):
        # -----------------------------
        # dimensions + init + bounds
        # -----------------------------
        if hard:
            q_dim = 4
        else:
            q_dim = 6 if enforce_symmetry else 8

        q0 = np.zeros(q_dim)

        a_min = float(getattr(config, "mu_logmag_min", -0.7))
        a_max = float(getattr(config, "mu_logmag_max",  0.7))
        phi_max = float(getattr(config, "mu_phase_max", np.pi))

        a_min_c = float(getattr(config, "mu_cpl_logmag_min", a_min))
        a_max_c = float(getattr(config, "mu_cpl_logmag_max", a_max))
        phi_max_c = float(getattr(config, "mu_cpl_phase_max", phi_max))

        lower = np.zeros(q_dim)
        upper = np.zeros(q_dim)

        # -----------------------------
        # diagonal bounds
        # -----------------------------
        lower[:4:2] = a_min
        upper[:4:2] = a_max
        lower[1:4:2] = -phi_max
        upper[1:4:2] =  phi_max

        # -----------------------------
        # NEW: diagonal initialization
        # -----------------------------
        if init_mu11 is not None:
            a11, phi11 = complex_to_q(init_mu11)
            q0[0] = np.clip(a11, a_min, a_max)
            q0[1] = np.clip(phi11, -phi_max, phi_max)

        if init_mu22 is not None:
            a22, phi22 = complex_to_q(init_mu22)
            q0[2] = np.clip(a22, a_min, a_max)
            q0[3] = np.clip(phi22, -phi_max, phi_max)

        # -----------------------------
        # coupling init (soft only)
        # -----------------------------
        if not hard:
            a12, phi12 = complex_to_q(mu12_prior)
            q0[4:6] = [a12, phi12]

            if not enforce_symmetry:
                a21, phi21 = complex_to_q(mu21_prior)
                q0[6:8] = [a21, phi21]

            lower[4::2] = a_min_c
            upper[4::2] = a_max_c
            lower[5::2] = -phi_max_c
            upper[5::2] =  phi_max_c
        # Build p_init prior from init_mu11/init_mu22
        p_init = None
        if (init_mu11 is not None) or (init_mu22 is not None):
            # Fallback: if one init is missing, use 1+0j for that diagonal
            mu11_0 = init_mu11 if init_mu11 is not None else (1.0 + 0.0j)
            mu22_0 = init_mu22 if init_mu22 is not None else (1.0 + 0.0j)

            # Define coupling prior consistent with hard/soft mode
            if hard:
                # hard mode coupling is implicit; p has 6 entries if symmetry else 8,
                # but we only care about diagonals in the prior
                if enforce_symmetry:
                    p_init = np.array([mu11_0.real, mu11_0.imag, mu22_0.real, mu22_0.imag, 0.0, 0.0], dtype=float)
                else:
                    p_init = np.array([mu11_0.real, mu11_0.imag, mu22_0.real, mu22_0.imag, 0.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                # soft mode: keep coupling near prior mu12_prior/mu21_prior (or set from init if you prefer)
                mu12_0 = mu12_prior
                mu21_0 = mu21_prior
                if enforce_symmetry:
                    p_init = np.array([mu11_0.real, mu11_0.imag, mu22_0.real, mu22_0.imag, mu12_0.real, mu12_0.imag], dtype=float)
                else:
                    p_init = np.array([mu11_0.real, mu11_0.imag, mu22_0.real, mu22_0.imag,
                                    mu12_0.real, mu12_0.imag, mu21_0.real, mu21_0.imag], dtype=float)

        # -----------------------------
        # residual (UNCHANGED)
        # -----------------------------
        def residual(q):
            p = q_to_p(
                q,
                enforce_symmetry,
                hard_constraint=hard,
                sign=sign,
            )

            # -----------------------------
            # penalty for negative real μ
            # -----------------------------
            reg_neg = np.empty(0)
            lam_neg = float(getattr(config, "mu_neg_real_lambda", 0.0))

            if lam_neg > 0:
                if enforce_symmetry:
                    mur11, _, mur22, _, mur12, _ = p
                    neg_terms = [
                        max(0.0, -mur11),
                        max(0.0, -mur22),
                    ]
                else:
                    mur11, _, mur22, _, mur12, _, mur21, _ = p
                    neg_terms = [
                        max(0.0, -mur11),
                        max(0.0, -mur22),
                    ]

                reg_neg = np.sqrt(lam_neg) * np.array(neg_terms, dtype=float)

            # -----------------------------
            # physics residual
            # -----------------------------
            mu_vec = build_mu_from_p_phys(p, enforce_symmetry)
            Rv = M_scaled @ (S12 * mu_vec) - b_2d
            Rv /= np.sqrt(Rv.size)

            # -----------------------------
            # regularization
            # -----------------------------
            lam_target = float(getattr(config, "mu_reg_lambda", 0.0))   # toward [1,0,1,0,...]
            lam_init   = float(getattr(config, "mu_init_lambda", 0.0))  # toward linear μ

            if enforce_symmetry:
                p_target = np.array([1, 0, 1, 0, 1, 0], dtype=float)
            else:
                p_target = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)

            W = build_mu_weight_vector(enforce_symmetry)
            reg_list = []

            if lam_target > 0:
                if hard:
                    reg_list.append(
                        np.sqrt(lam_target) * np.sqrt(W[:4]) * (p[:4] - p_target[:4])
                    )
                else:
                    reg_list.append(
                        np.sqrt(lam_target) * np.sqrt(W) * (p - p_target)
                    )

            if lam_init > 0 and (p_init is not None):
                if hard:
                    reg_list.append(
                        np.sqrt(lam_init) * np.sqrt(W[:4]) * (p[:4] - p_init[:4])
                    )
                else:
                    reg_list.append(
                        np.sqrt(lam_init) * np.sqrt(W) * (p - p_init)
                    )

            reg = np.concatenate(reg_list) if reg_list else np.empty(0)

            # -----------------------------
            # algebraic constraint (soft mode only)
            # -----------------------------
            reg_alg = np.empty(0)
            if not hard:
                lam_c = float(getattr(config, "mu_constraint_lambda", 0.0))
                if lam_c > 0:
                    mu11, mu22, mu12, _ = q_to_mu_complex(
                        q, enforce_symmetry, hard_constraint=False, sign=sign
                    )
                    c = mu12**2 - mu11 * mu22
                    reg_alg = np.sqrt(lam_c) * np.array([c.real, c.imag], dtype=float)

            # -----------------------------
            # combine all residuals
            # -----------------------------
            res = np.concatenate([Rv, reg, reg_alg, reg_neg])

            bad = ~np.isfinite(res)
            res[bad] = 1e6
            return res
        
        res = least_squares(
            residual,
            q0,
            method=getattr(config, "lsq_method", "trf"),
            bounds=(lower, upper),
        )

        q_opt = res.x
        p_opt = q_to_p(
            q_opt,
            enforce_symmetry,
            hard_constraint=hard,
            sign=sign,
        )

        r = residual(q_opt)
        r = r[np.isfinite(r)]
        rn = float(np.linalg.norm(r)) if r.size > 0 else np.nan

        return p_opt, res.cost, rn

    def summarize(branch):
        if quiet:
            return {}
        vals = [d for d in block_info if d["block"] == branch]
        if not vals:
            return {}
        return dict(
            rank_scaled_min=int(np.min([v["rank_scaled"] for v in vals])),
            rank_scaled_max=int(np.max([v["rank_scaled"] for v in vals])),
            svd_sigma_min_scaled_min=float(np.min([v["svd_sigma_min_scaled"] for v in vals])),
            svd_ratio_scaled_min=float(np.min([v["svd_ratio_scaled"] for v in vals])),
            cond_scaled_min=float(np.min([v["cond_scaled"] for v in vals])),
            cond_scaled_max=float(np.max([v["cond_scaled"] for v in vals])),
            num_blocks=len(vals),
        )

    # =========================================================
    # run solver(s)
    # =========================================================
    
    if hard:
        p_p, cost_p, rn_p = solve_for_sign(+1)
        p_m, cost_m, rn_m = solve_for_sign(-1)

        if cost_p <= cost_m:
            p_opt, best_cost, best_rn, best_sign = p_p, cost_p, rn_p, +1
        else:
            p_opt, best_cost, best_rn, best_sign = p_m, cost_m, rn_m, -1

        info = dict(
            global_cost=best_cost,
            residual_norm=best_rn,

            cond_M=cond_safe(M_2d),
            cond_M_scaled=cond_safe(M_scaled),
            rank_svd=svd_rank(M_scaled, rel_thresh=rel_svd_thresh)[0],
            mu_svd_rel_thresh=rel_svd_thresh,

            branch1_summary=summarize("branch1"),
            branch2_summary=summarize("branch2"),

            p_dim=(6 if enforce_symmetry else 8),
            mu12_sign=best_sign,
        )

        if not quiet:
            info["block_info"] = block_info

    else:
        p_opt, best_cost, best_rn = solve_for_sign(+1)
        info = dict(
            global_cost=best_cost,
            residual_norm=best_rn,

            cond_M=cond_safe(M_2d),
            cond_M_scaled=cond_safe(M_scaled),
            rank_svd=svd_rank(M_scaled, rel_thresh=rel_svd_thresh)[0],
            mu_svd_rel_thresh=rel_svd_thresh,

            branch1_summary=summarize("branch1"),
            branch2_summary=summarize("branch2"),

            p_dim=(6 if enforce_symmetry else 8),
        )

        if not quiet:
            info["block_info"] = block_info

    return p_opt, info
