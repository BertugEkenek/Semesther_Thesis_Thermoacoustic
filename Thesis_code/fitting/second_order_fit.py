# fitting/second_order_fit.py

import numpy as np
from scipy.optimize import least_squares
from utils import logger

# ============================================================
#          SECOND-ORDER μ-FIT — CLEAN GLOBAL PER-R SOLVER
# ============================================================
# ============================================================
# Strategy dispatch (single knob)
# ============================================================

# Canonical strategy names + backward aliases
STRATEGY_ALIASES = {
    # old / experimental names -> canonical
    "hybrid_linear_diag_analytic_coupling": "hybrid_analytic_mu12_from_diag",
}

# Strategy families (base strategies are already in STRATEGIES)
HYBRID_STRATEGIES = {
    "hybrid_analytic_mu12_from_diag",
    "hybrid_freeze_diag_rank1_mag_phasefree",
    "hybrid_freeze_diag_sym_coupling",
    "hybrid_freeze_diag_rank1_mag_antiphase"
}

def resolve_mu_strategy(config) -> str:
    """
    Resolve the single knob for μ fitting.

    Preferred: config.mu_strategy
    Backward compatible: config.mu_fit_strategy
    Applies STRATEGY_ALIASES.
    """
    name = getattr(config, "mu_strategy", None)
    if name is None:
        name = getattr(config, "mu_fit_strategy", None)
    if name is None:
        # keep explicit so failures are clean
        raise AttributeError("Set config.mu_strategy (preferred) or config.mu_fit_strategy.")

    name = STRATEGY_ALIASES.get(name, name)
    return str(name)

def _sqrt_with_pos_real(z: complex) -> complex:
    """
    Return sqrt(z) with Re(root) >= 0 (ties broken by Im >= 0).
    """
    r = np.sqrt(z)
    if (r.real < 0) or (np.isclose(r.real, 0.0) and r.imag < 0):
        r = -r
    return r

def _format_c(mu: complex) -> str:
    return f"{mu.real:.6f} + i {mu.imag:.6f}"


def _extract_mu_dict_from_p(p):
    """
    Returns a dict with complex μ entries inferred from p.
    Supports p.size == 6 or 8.
    """
    p = np.asarray(p, float).ravel()

    if p.size == 6:
        mu11 = p[0] + 1j * p[1]
        mu22 = p[2] + 1j * p[3]
        mu12 = p[4] + 1j * p[5]
        mu21 = mu12
    elif p.size == 8:
        mu11 = p[0] + 1j * p[1]
        mu22 = p[2] + 1j * p[3]
        mu12 = p[4] + 1j * p[5]
        mu21 = p[6] + 1j * p[7]
    else:
        raise ValueError(f"Expected p.size in {{6,8}}, got {p.size}")

    return {
        "mu11": mu11,
        "mu22": mu22,
        "mu12": mu12,
        "mu21": mu21,
    }
# -----------------------------
# numerics helpers
# -----------------------------
def cond_safe(A) -> float:
    try:
        return float(np.linalg.cond(A))
    except Exception:
        return float(np.inf)


def svd_rank(A, rel_thresh=1e-10):
    """
    Returns (rank, sigma_max, sigma_min, sigma_min/sigma_max, svals)
    using a relative threshold on sigma_max.
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


def complex_to_q(mu: complex):
    """Map complex μ to (log|μ|, arg(μ))."""
    r = max(abs(mu), 1e-12)
    return np.log(r), np.arctan2(mu.imag, mu.real)


def _cplx_from_a_phi(a, phi):
    return np.exp(a) * (np.cos(phi) + 1j * np.sin(phi))


# ============================================================
# Strategies (config.mu_fit_strategy)
# ============================================================

def q_to_p_rank1_sym(q, sign=+1):
    # q = [a11, ph11, a22, ph22], coupling implied, mu12=mu21, sign scanned
    a11, ph11, a22, ph22 = q
    mu11 = _cplx_from_a_phi(a11, ph11)
    mu22 = _cplx_from_a_phi(a22, ph22)
    a12 = 0.5 * (a11 + a22)
    ph12 = 0.5 * (ph11 + ph22)
    mu12 = sign * _cplx_from_a_phi(a12, ph12)
    return np.array([mu11.real, mu11.imag, mu22.real, mu22.imag, mu12.real, mu12.imag], float)


def q_to_p_sym_only(q):
    # q = [a11, ph11, a22, ph22, a12, ph12], mu21=mu12
    a11, ph11, a22, ph22, a12, ph12 = q
    mu11 = _cplx_from_a_phi(a11, ph11)
    mu22 = _cplx_from_a_phi(a22, ph22)
    mu12 = _cplx_from_a_phi(a12, ph12)
    return np.array([mu11.real, mu11.imag, mu22.real, mu22.imag, mu12.real, mu12.imag], float)


def q_to_p_none(q):
    # q = [a11, ph11, a22, ph22, a12, ph12, a21, ph21]
    a11, ph11, a22, ph22, a12, ph12, a21, ph21 = q
    mu11 = _cplx_from_a_phi(a11, ph11)
    mu22 = _cplx_from_a_phi(a22, ph22)
    mu12 = _cplx_from_a_phi(a12, ph12)
    mu21 = _cplx_from_a_phi(a21, ph21)
    return np.array(
        [mu11.real, mu11.imag, mu22.real, mu22.imag, mu12.real, mu12.imag, mu21.real, mu21.imag],
        float
    )


def q_to_p_rank1_mag_phasefree(q):
    # q = [a11, ph11, a22, ph22, ph12, ph21], coupling magnitude implied, phases free
    a11, ph11, a22, ph22, ph12, ph21 = q
    mu11 = _cplx_from_a_phi(a11, ph11)
    mu22 = _cplx_from_a_phi(a22, ph22)

    a12 = 0.5 * (a11 + a22)  # shared coupling log-mag
    mu12 = _cplx_from_a_phi(a12, ph12)
    mu21 = _cplx_from_a_phi(a12, ph21)

    return np.array(
        [mu11.real, mu11.imag, mu22.real, mu22.imag, mu12.real, mu12.imag, mu21.real, mu21.imag],
        float
    )
def q_to_p_rank1_mag_antiphase(q):
    # q = [a11, ph11, a22, ph22, ph] with |mu12|=|mu21| implied, and ph21=-ph12
    a11, ph11, a22, ph22, ph = q
    mu11 = _cplx_from_a_phi(a11, ph11)
    mu22 = _cplx_from_a_phi(a22, ph22)

    a12 = 0.5 * (a11 + a22)
    mu12 = _cplx_from_a_phi(a12,  ph)
    mu21 = _cplx_from_a_phi(a12, -ph)

    return np.array(
        [mu11.real, mu11.imag, mu22.real, mu22.imag, mu12.real, mu12.imag, mu21.real, mu21.imag],
        float
    )

STRATEGIES = {
    "rank1_sym": dict(q_dim=4, symmetric=True,  sign_scan=True),
    "rank1_mag_phasefree": dict(q_dim=6, symmetric=False, sign_scan=False),
    "rank1_mag_antiphase": dict(q_dim=5, symmetric=False, sign_scan=False),
    "none": dict(q_dim=8, symmetric=False, sign_scan=False),
    "sym_only": dict(q_dim=6, symmetric=True, sign_scan=False),
}


def p_to_q_from_previous(p_prev, strategy_name: str):
    p_prev = np.asarray(p_prev, float).ravel()

    if strategy_name == "rank1_sym":
        mur11, mui11, mur22, mui22 = p_prev[:4]
        a11, ph11 = complex_to_q(mur11 + 1j*mui11)
        a22, ph22 = complex_to_q(mur22 + 1j*mui22)
        return np.array([a11, ph11, a22, ph22], float)

    if strategy_name == "sym_only":
        mur11, mui11, mur22, mui22, mur12, mui12 = p_prev[:6]
        a11, ph11 = complex_to_q(mur11 + 1j*mui11)
        a22, ph22 = complex_to_q(mur22 + 1j*mui22)
        a12, ph12 = complex_to_q(mur12 + 1j*mui12)
        return np.array([a11, ph11, a22, ph22, a12, ph12], float)

    if strategy_name == "none":
        mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21 = p_prev[:8]
        a11, ph11 = complex_to_q(mur11 + 1j*mui11)
        a22, ph22 = complex_to_q(mur22 + 1j*mui22)
        a12, ph12 = complex_to_q(mur12 + 1j*mui12)
        a21, ph21 = complex_to_q(mur21 + 1j*mui21)
        return np.array([a11, ph11, a22, ph22, a12, ph12, a21, ph21], float)

    if strategy_name == "rank1_mag_phasefree":
        # only phases are free; coupling magnitude implied from diagonals
        mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21 = p_prev[:8]
        a11, ph11 = complex_to_q(mur11 + 1j*mui11)
        a22, ph22 = complex_to_q(mur22 + 1j*mui22)
        ph12 = np.arctan2(mui12, mur12)
        ph21 = np.arctan2(mui21, mur21)
        return np.array([a11, ph11, a22, ph22, ph12, ph21], float)
    
    if strategy_name == "rank1_mag_antiphase":
        mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21 = p_prev[:8]
        a11, ph11 = complex_to_q(mur11 + 1j*mui11)
        a22, ph22 = complex_to_q(mur22 + 1j*mui22)

        ph12 = np.arctan2(mui12, mur12)
        ph21 = np.arctan2(mui21, mur21)

        # project onto anti-phase manifold: ph21 ≈ -ph12
        ph = 0.5 * (ph12 - ph21)
        return np.array([a11, ph11, a22, ph22, ph], float)

    raise ValueError(f"Unknown strategy_name={strategy_name}")
# -----------------------------
# normalized terms (stable)
# -----------------------------
def compute_normalized_terms(sigma, w, s_ref, s_1, s_2):
    """
    Compute normalized ratios:
      ratio_s = s^2 / s_ref^2  (vector)
      ratio_1 = s1^2 / s_ref^2 (scalar)
      ratio_2 = s2^2 / s_ref^2 (scalar)

    Returns:
      (Crn, Cin, Cr1, Ci1, Cr2, Ci2)
    """
    sigma = np.asarray(sigma, float)
    w = np.asarray(w, float)

    s = sigma + 1j * w

    s_ref2 = complex(s_ref) ** 2
    s1_2 = complex(s_1) ** 2
    s2_2 = complex(s_2) ** 2

    eps = 1e-12
    if abs(s_ref2) < eps:
        logger.warning(f"[compute_normalized_terms] |s_ref^2| too small. Using eps. s_ref={s_ref}")
        s_ref2 = eps

    ratio_s = (s * s) / s_ref2
    ratio_1 = s1_2 / s_ref2
    ratio_2 = s2_2 / s_ref2

    return (
        ratio_s.real, ratio_s.imag,
        float(ratio_1.real), float(ratio_1.imag),
        float(ratio_2.real), float(ratio_2.imag),
    )


# ============================================================
#  CORE: build M and b for one trajectory (vectorized in n)
# ============================================================
def _build_M_all_and_b_all(config, tau, w, sigma, s_ref, s_1, s_2, n):
    """
    Returns:
      M_all: (m, 2, 12)  [Re, Im] rows for each n-sample
      b_all: (m, 2)
    """
    m = len(n) - 1  # exclude n=0 sample which is not used in fitting
    idx = np.arange(m)

    alphaK1 = config.alpha[0] * config.K
    alphaK2 = config.alpha[1] * config.K

    exp_term = np.exp(-tau * sigma[idx])

    C11 = config.Lambda[0] * alphaK1 * config.nu[0] * n[1:] * exp_term
    C22 = config.Lambda[1] * alphaK2 * config.nu[1] * n[1:] * exp_term
    C12 = config.Lambda[1] * alphaK2 * config.nu[2] * n[1:] * exp_term
    C21 = config.Lambda[0] * alphaK1 * config.nu[3] * n[1:] * exp_term

    Crn, Cin, Cr1, Ci1, Cr2, Ci2 = compute_normalized_terms(
        sigma[idx], w[idx], s_ref, s_1, s_2
    )

    s = np.sin(w[idx] * tau)
    c = np.cos(w[idx] * tau)

    M_all = np.zeros((m, 2, 12), float)
    b_all = np.zeros((m, 2), float)

    # --- Re row (row 0) ---
    M_all[:, 0, 0] = C11*(Cin**2*c - Crn**2*c - Ci2*Cin*c + Cr2*Crn*c +
                         Ci2*Crn*s + Cin*Cr2*s - 2*Cin*Crn*s)
    M_all[:, 0, 1] = -C11*(Crn**2*s - Cin**2*s + Ci2*Crn*c + Cin*Cr2*c -
                           2*Cin*Crn*c + Ci2*Cin*s - Cr2*Crn*s)

    M_all[:, 0, 2] = C22*(Cin**2*c - Crn**2*c - Ci1*Cin*c + Cr1*Crn*c +
                          Ci1*Crn*s + Cin*Cr1*s - 2*Cin*Crn*s)
    M_all[:, 0, 3] = -C22*(Crn**2*s - Cin**2*s + Ci1*Crn*c + Cin*Cr1*c -
                           2*Cin*Crn*c + Ci1*Cin*s - Cr1*Crn*s)

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

    # --- Im row (row 1) ---
    M_all[:, 1, 0] = C11*(Crn**2*s - Cin**2*s + Ci2*Crn*c +
                         Cin*Cr2*c - 2*Cin*Crn*c +
                         Ci2*Cin*s - Cr2*Crn*s)
    M_all[:, 1, 1] = C11*(Cin**2*c - Crn**2*c - Ci2*Cin*c +
                         Cr2*Crn*c + Ci2*Crn*s +
                         Cin*Cr2*s - 2*Cin*Crn*s)

    M_all[:, 1, 2] = C22*(Crn**2*s - Cin**2*s + Ci1*Crn*c +
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
    M_all[:, 1, 11] = - M_all[:, 1, 8]

    # --- RHS constants moved to b ---
    b_all[:, 0] = -(
        -Ci1*Ci2 + Ci1*Cin + Ci2*Cin - Cin**2
        + Cr1*Cr2 - Cr1*Crn - Cr2*Crn + Crn**2
    )
    b_all[:, 1] = -(
        Ci1*Cr2 - Ci1*Crn + Ci2*Cr1 - Ci2*Crn
        - Cin*Cr1 - Cin*Cr2 + 2*Cin*Crn
    )

    return M_all, b_all


def _collapse_12_to_identifiable(M_all, config):
    """
    Collapse 12 → 8 identifiable groups (always).
    Returns:
      A_phys: (N, 8)
    """
    m = M_all.shape[0]
    M8 = np.zeros((m, 2, 8), float)
    M8[:, :, 0:4] = M_all[:, :, 0:4]
    M8[:, :, 4] = 0.5 * (M_all[:, :, 4] - M_all[:, :, 7])   # Re(mu11*mu22)
    M8[:, :, 5] = 0.5 * (M_all[:, :, 5] + M_all[:, :, 6])   # Im(mu11*mu22)
    M8[:, :, 6] = 0.5 * (M_all[:, :, 8] - M_all[:, :, 11])  # Re(mu12*mu21)
    M8[:, :, 7] = 0.5 * (M_all[:, :, 9] + M_all[:, :, 10])  # Im(mu12*mu21)
    return M8.reshape(-1, 8)


# ============================================================
#  p -> identifiable mu vec (8)
# ============================================================
def build_mu_from_p_phys_collapsed(p):
    """
    Build identifiable μ-vector of length 8 used by A_phys:
      [Re11, Im11, Re22, Im22, Re(mu11*mu22), Im(mu11*mu22), Re(mu12*mu21), Im(mu12*mu21)]

    If p is length 6 -> assumes mu21=mu12.
    If p is length 8 -> full.
    """
    p = np.asarray(p, float).ravel()

    if p.size == 6:
        mur11, mui11, mur22, mui22, mur12, mui12 = p
        mur21, mui21 = mur12, mui12
    elif p.size == 8:
        mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21 = p
    else:
        raise ValueError(f"p must be length 6 or 8, got {p.size}")

    # products:
    re_d = mur11 * mur22 - mui11 * mui22
    im_d = mur11 * mui22 + mui11 * mur22

    re_c = mur12 * mur21 - mui12 * mui21
    im_c = mur12 * mui21 + mui12 * mur21

    return np.array([mur11, mui11, mur22, mui22, re_d, im_d, re_c, im_c], float)


# ============================================================
#  solve per R (global stacked nonlinear LSQ)
# ============================================================
def solve_mu_per_R(
    config,
    n,
    data_blocks,
    quiet=False,
    init_mu11: complex | None = None,
    init_mu22: complex | None = None,
    prev_p_opt=None,
):
    """
    Solve one global second-order μ-fit for a single R.

    Returns:
      p_opt, info
    where p_opt is (6 or 8)-vector of real μ parameters.

    data_blocks: list of dicts, each containing:
      {
        'tag': str,
        'tau': float,
        'w': array (m,),
        'sigma': array (m,),
        's_ref': complex,
        's_1': complex,
        's_2': complex
      }
    """
    # -----------------------------
    # strategy selection
    # -----------------------------
    strategy_name = getattr(config, "mu_fit_strategy", "none")
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown mu_fit_strategy='{strategy_name}'. Available: {list(STRATEGIES.keys())}")

    S = STRATEGIES[strategy_name]
    q_dim = int(S["q_dim"])
    is_sym = bool(S["symmetric"])
    sign_scan = bool(S["sign_scan"])


    # -----------------------------
    # build stacked physics system
    # -----------------------------
    A_blocks = []
    b_blocks = []
    block_info = []

    rel_col_thresh = float(getattr(config, "mu_col_rel_thresh", 1e-3))
    rel_svd_thresh = float(getattr(config, "mu_svd_rel_thresh", 1e-10))

    def _append_block(tag, w_blk, sig_blk, s_ref_blk, tau_blk, s_1_blk, s_2_blk):
        w_blk = np.asarray(w_blk).reshape(-1)
        sig_blk = np.asarray(sig_blk).reshape(-1)

        if w_blk.shape != sig_blk.shape:
            raise ValueError(f"{tag}: w_blk and sig_blk must have same shape, got {w_blk.shape} vs {sig_blk.shape}")

        M_all, b_all = _build_M_all_and_b_all(
            config, tau_blk, w_blk, sig_blk, s_ref_blk, s_1_blk, s_2_blk, n
        )
        A_phys_blk = _collapse_12_to_identifiable(M_all, config)
        b_2d_blk = b_all.reshape(-1)

        # Column scaling
        col_norms = np.linalg.norm(A_phys_blk, axis=0)
        max_norm = np.max(col_norms) if np.any(col_norms > 0) else 1.0
        eff = np.maximum(col_norms, rel_col_thresh * max_norm)
        A_scaled = A_phys_blk / eff

        A_blocks.append((A_scaled, eff, A_phys_blk))
        b_blocks.append(b_2d_blk)

        rnk, smax, smin, ratio, _ = svd_rank(A_scaled, rel_thresh=rel_svd_thresh)
        block_info.append(
            dict(
                block=tag,
                cond_scaled=cond_safe(A_scaled),
                rank_scaled=rnk,
                svd_sigma_min_scaled=smin,
                svd_ratio_scaled=ratio,
                p_dim=A_phys_blk.shape[1],
            )
        )

    for blk in data_blocks:
        _append_block(
            tag=blk["tag"],
            w_blk=blk["w"],
            sig_blk=blk["sigma"],
            s_ref_blk=blk["s_ref"],
            tau_blk=blk["tau"],
            s_1_blk=blk["s_1"],
            s_2_blk=blk["s_2"],
        )

    A_phys = np.vstack([x[2] for x in A_blocks])
    b_2d = np.concatenate(b_blocks)

    # -----------------------------
    # q-dimension + bounds + init
    # -----------------------------
    q0 = np.zeros(q_dim)

    a_min = float(getattr(config, "mu_logmag_min", -0.7))
    a_max = float(getattr(config, "mu_logmag_max",  0.7))
    phi_max = float(getattr(config, "mu_phase_max", np.pi))

    a_min_c = float(getattr(config, "mu_cpl_logmag_min", a_min))
    a_max_c = float(getattr(config, "mu_cpl_logmag_max", a_max))
    phi_max_c = float(getattr(config, "mu_cpl_phase_max", phi_max))

    lower = np.full(q_dim, -np.inf)
    upper = np.full(q_dim, +np.inf)

    # -------------------------------------------------
    # 1) bounds: diagonals always exist in indices 0..3
    # -------------------------------------------------
    lower[:4:2] = a_min
    upper[:4:2] = a_max
    lower[1:4:2] = -phi_max
    upper[1:4:2] = +phi_max

    # -------------------------------------------------
    # 2) bounds: coupling (MUST be set before warm-start)
    # -------------------------------------------------
    mu12_prior = getattr(config, "mu12_prior", 1.0 + 0.0j)
    mu21_prior = getattr(config, "mu21_prior", mu12_prior)

    if strategy_name in ("sym_only", "none"):
        lower[4:6] = [a_min_c, -phi_max_c]
        upper[4:6] = [a_max_c, +phi_max_c]
        if strategy_name == "none":
            lower[6:8] = [a_min_c, -phi_max_c]
            upper[6:8] = [a_max_c, +phi_max_c]

    elif strategy_name == "rank1_mag_phasefree":
        # q layout: [a11,ph11,a22,ph22,ph12,ph21]
        lower[4:6] = [-phi_max_c, -phi_max_c]
        upper[4:6] = [+phi_max_c, +phi_max_c]
    elif strategy_name == "rank1_mag_antiphase":
    # q layout: [a11,ph11,a22,ph22,ph] with ph21=-ph12
        lower[4] = -phi_max_c
        upper[4] = +phi_max_c
    elif strategy_name == "rank1_sym":
        # q layout: [a11,ph11,a22,ph22]
        pass
    else:
        raise RuntimeError(f"Internal: unknown strategy_name '{strategy_name}'")

    # -------------------------------------------------
    # 3) init: diagonals from linear init (optional)
    # -------------------------------------------------
    if init_mu11 is not None:
        a11, ph11 = complex_to_q(init_mu11)
        q0[0] = np.clip(a11, a_min, a_max)
        q0[1] = np.clip(ph11, -phi_max, phi_max)

    if init_mu22 is not None:
        a22, ph22 = complex_to_q(init_mu22)
        q0[2] = np.clip(a22, a_min, a_max)
        q0[3] = np.clip(ph22, -phi_max, phi_max)

    # -------------------------------------------------
    # 4) warm start from previous R (clip against FINAL bounds)
    # -------------------------------------------------
    did_warm_start = False
    if prev_p_opt is not None:
        try:
            q_prev = p_to_q_from_previous(prev_p_opt, strategy_name=strategy_name)
            k = min(q_prev.size, q0.size)
            q0[:k] = np.clip(q_prev[:k], lower[:k], upper[:k])
            did_warm_start = True
            if not quiet:
                logger.info("[Second-order μ | warm-start] Using previous R solution.")
        except Exception as e:
            logger.warning(f"[Second-order μ | warm-start] Failed, fallback to init/prior. Reason: {e}")

    # -------------------------------------------------
    # 5) if no warm-start, seed coupling from priors
    #    (do NOT overwrite warm-started coupling)
    # -------------------------------------------------
    if not did_warm_start:
        if strategy_name in ("sym_only", "none"):
            a12, ph12 = complex_to_q(mu12_prior)
            q0[4] = np.clip(a12, a_min_c, a_max_c)
            q0[5] = np.clip(ph12, -phi_max_c, phi_max_c)

            if strategy_name == "none":
                a21, ph21 = complex_to_q(mu21_prior)
                q0[6] = np.clip(a21, a_min_c, a_max_c)
                q0[7] = np.clip(ph21, -phi_max_c, phi_max_c)

        elif strategy_name == "rank1_mag_phasefree":
            q0[4] = np.clip(np.angle(mu12_prior), -phi_max_c, phi_max_c)
            q0[5] = np.clip(np.angle(mu21_prior), -phi_max_c, phi_max_c)
        elif strategy_name == "rank1_mag_antiphase":
            # q layout: [a11,ph11,a22,ph22,ph] with ph21=-ph12
            lower[4] = -phi_max_c
            upper[4] = +phi_max_c

    # rank1_sym: nothing to seed
    # -----------------------------
    # optional regularization targets
    # -----------------------------
    p_init = None
    p_cont = None
    if prev_p_opt is not None:
        p_cont = np.asarray(prev_p_opt, float).ravel()

    if (init_mu11 is not None) or (init_mu22 is not None):
        mu11_0 = init_mu11 if init_mu11 is not None else (1.0 + 0.0j)
        mu22_0 = init_mu22 if init_mu22 is not None else (1.0 + 0.0j)
        if is_sym:
            p_init = np.array(
                [mu11_0.real, mu11_0.imag, mu22_0.real, mu22_0.imag, mu12_prior.real, mu12_prior.imag],
                float,
            )
        else:
            p_init = np.array(
                [mu11_0.real, mu11_0.imag, mu22_0.real, mu22_0.imag,
                 mu12_prior.real, mu12_prior.imag, mu21_prior.real, mu21_prior.imag],
                float,
            )

    lam_target = float(getattr(config, "mu_modelIII_lambda", 0.0))
    lam_init   = float(getattr(config, "mu_init_lambda", 0.0))
    lam_one    = float(getattr(config, "mu_one_target_lambda", 0.0))
    lam_cont   = float(getattr(config, "mu_continuation_lambda", 0.0))

    p_target = np.array([1, 0, 1, 0, 1, 0], float) if is_sym else np.array([1, 0, 1, 0, 1, 0, 1, 0], float)

    # -----------------------------
    # solver (sign scan only for rank1_sym)
    # -----------------------------
    def solve_for_sign(sign):
        def residual(q):
            if strategy_name == "rank1_sym":
                p = q_to_p_rank1_sym(q, sign=sign)
            elif strategy_name == "sym_only":
                p = q_to_p_sym_only(q)
            elif strategy_name == "none":
                p = q_to_p_none(q)
            elif strategy_name == "rank1_mag_phasefree":
                p = q_to_p_rank1_mag_phasefree(q)
            elif strategy_name == "rank1_mag_antiphase":
                p = q_to_p_rank1_mag_antiphase(q)
            else:
                raise RuntimeError("Internal: unknown strategy dispatch")

            mu_vec = build_mu_from_p_phys_collapsed(p)  # always (8,)
            r_phys = A_phys @ mu_vec - b_2d
            Rv = r_phys / np.sqrt(max(r_phys.size, 1))

            reg_list = []

            if lam_target > 0:
                reg_list.append(np.sqrt(lam_target) * (p - p_target))

            if lam_init > 0 and (p_init is not None):
                reg_list.append(np.sqrt(lam_init) * (p - p_init))

            if lam_cont > 0 and (p_cont is not None):
                # continuation can be mismatched dim if strategy differs across runs; guard it
                if p_cont.size == p.size:
                    reg_list.append(np.sqrt(lam_cont) * (p - p_cont))

            if lam_one > 0:
                # penalize Re(mu11) and Re(mu22) toward 1.0 (dimension-agnostic)
                mur11 = float(p[0])
                mur22 = float(p[2])
                reg_real = np.sqrt(lam_one) * np.array([mur11 - 1.0, mur22 - 1.0], float)
                reg_list.append(reg_real)

            reg = np.concatenate(reg_list) if reg_list else np.empty(0)
            out = np.concatenate([Rv, reg])

            bad = ~np.isfinite(out)
            out[bad] = 1e6
            return out

        method = getattr(config, "lsq_method", "trf")
        if method == "trf":
            res = least_squares(residual, q0, method=method, bounds=(lower, upper))
        else:
            res = least_squares(residual, q0, method=method)

        q_opt = res.x

        # reconstruct p
        if strategy_name == "rank1_sym":
            p_opt = q_to_p_rank1_sym(q_opt, sign=sign)
        elif strategy_name == "sym_only":
            p_opt = q_to_p_sym_only(q_opt)
        elif strategy_name == "none":
            p_opt = q_to_p_none(q_opt)
        elif strategy_name == "rank1_mag_phasefree":
            p_opt = q_to_p_rank1_mag_phasefree(q_opt)
        elif strategy_name == "rank1_mag_antiphase":
            p_opt = q_to_p_rank1_mag_antiphase(q_opt)
        else:
            raise RuntimeError("Internal: unknown strategy dispatch")

        r = residual(q_opt)
        r = r[np.isfinite(r)]
        rn = float(np.linalg.norm(r)) if r.size else np.nan
        return p_opt, float(res.cost), rn

    if sign_scan:
        p_p, cost_p, rn_p = solve_for_sign(+1)
        p_m, cost_m, rn_m = solve_for_sign(-1)
        if cost_p <= cost_m:
            p_opt, best_cost, best_rn, best_sign = p_p, cost_p, rn_p, +1
        else:
            p_opt, best_cost, best_rn, best_sign = p_m, cost_m, rn_m, -1
    else:
        p_opt, best_cost, best_rn = solve_for_sign(+1)
        best_sign = +1

    mu_vec = build_mu_from_p_phys_collapsed(p_opt)
    r_phys = A_phys @ mu_vec - b_2d
    N_phys = max(r_phys.size, 1)

    phys_cost = 0.5 * float(np.dot(r_phys, r_phys)) / N_phys
    phys_rms = float(np.linalg.norm(r_phys) / np.sqrt(N_phys))

    # -----------------------------
    # info summary
    # -----------------------------
    def summarize(tag):
        vals = [d for d in block_info if d["block"] == tag]
        if not vals:
            return {}
        ranks = [v["rank_scaled"] for v in vals if np.isfinite(v["rank_scaled"])]
        conds = [v["cond_scaled"] for v in vals if np.isfinite(v["cond_scaled"])]
        ratios = [v["svd_ratio_scaled"] for v in vals if np.isfinite(v["svd_ratio_scaled"])]
        return dict(
            rank_scaled_min=int(np.min(ranks)) if ranks else None,
            rank_scaled_max=int(np.max(ranks)) if ranks else None,
            cond_scaled_min=float(np.min(conds)) if conds else np.nan,
            cond_scaled_max=float(np.max(conds)) if conds else np.nan,
            svd_ratio_scaled_min=float(np.min(ratios)) if ratios else np.nan,
            num_blocks=len(vals),
        )
    mu_dict = _extract_mu_dict_from_p(p_opt)
    info = dict(
        mu_fit_strategy=strategy_name,
        global_cost=float(best_cost),
        residual_norm=float(best_rn),
        phys_cost=float(phys_cost),
        phys_rms=float(phys_rms),
        cond_A_phys=cond_safe(A_phys),
        rank_A_phys=svd_rank(A_phys, rel_thresh=rel_svd_thresh)[0],
        mu_svd_rel_thresh=rel_svd_thresh,
        p_dim=int(np.asarray(p_opt).size),
        mu12_sign=int(best_sign),
        used_warm_start=bool(did_warm_start),
        sign_scan=bool(sign_scan),
        A_phys_shape=A_phys.shape,
        mu11=mu_dict["mu11"],
        mu22=mu_dict["mu22"],
        mu12=mu_dict["mu12"],
        mu21=mu_dict["mu21"],
    )

    unique_tags = sorted(list(set(b["tag"] for b in data_blocks)))
    for tag in unique_tags:
        info[f"{tag}_summary"] = summarize(tag)

    if not quiet:
        prefix = f"[Second-order μ regression | strategy={strategy_name}]"
        logger.info(f"{prefix} --------------------------------")
        logger.info(f"{prefix} A_phys.shape          = {A_phys.shape}")
        logger.info(f"{prefix} μ11                  = {_format_c(mu_dict['mu11'])}")
        logger.info(f"{prefix} μ22                  = {_format_c(mu_dict['mu22'])}")
        logger.info(f"{prefix} μ12                  = {_format_c(mu_dict['mu12'])}")
        logger.info(f"{prefix} μ21                  = {_format_c(mu_dict['mu21'])}")
        logger.info(f"{prefix} global_cost          = {best_cost:.3e}")
        logger.info(f"{prefix} phys_cost            = {phys_cost:.3e}")
        logger.info(f"{prefix} phys_rms             = {phys_rms:.3e}")
        logger.info(f"{prefix} residual_norm        = {best_rn:.3e}")
        logger.info(f"{prefix} rank(A_phys)         = {info['rank_A_phys']}")
        logger.info(f"{prefix} cond(A_phys)         = {info['cond_A_phys']:.3e}")
        logger.info(f"{prefix} p_dim                = {info['p_dim']}")
        logger.info(f"{prefix} sign_scan            = {info['sign_scan']}")
        logger.info(f"{prefix} mu12_sign            = {info['mu12_sign']}")
        logger.info(f"{prefix} used_warm_start      = {info['used_warm_start']}")
        logger.info(f"{prefix} --------------------------------")
        info["block_info"] = block_info

    return p_opt, info

# ============================================================
#  HYBRID / FROZEN-DIAGONAL COUPLING-ONLY SOLVERS
# ============================================================

def _build_Aphys_and_b(config, n, data_blocks, rel_col_thresh=1e-3, rel_svd_thresh=1e-10):
    """
    Shared helper: build stacked A_phys (N,8) and b_2d (N,)
    consistent with solve_mu_per_R().
    """
    A_blocks = []
    b_blocks = []
    block_info = []

    def _append_block(tag, w_blk, sig_blk, s_ref_blk, tau_blk, s_1_blk, s_2_blk):
        w_blk = np.asarray(w_blk).reshape(-1)
        sig_blk = np.asarray(sig_blk).reshape(-1)
        if w_blk.shape != sig_blk.shape:
            raise ValueError(f"{tag}: w_blk and sig_blk must have same shape, got {w_blk.shape} vs {sig_blk.shape}")

        M_all, b_all = _build_M_all_and_b_all(
            config, tau_blk, w_blk, sig_blk, s_ref_blk, s_1_blk, s_2_blk, n
        )
        A_phys_blk = _collapse_12_to_identifiable(M_all, config)
        b_2d_blk = b_all.reshape(-1)

        # Column scaling diagnostics only (we solve on unscaled A_phys for consistency)
        col_norms = np.linalg.norm(A_phys_blk, axis=0)
        max_norm = np.max(col_norms) if np.any(col_norms > 0) else 1.0
        eff = np.maximum(col_norms, rel_col_thresh * max_norm)
        A_scaled = A_phys_blk / eff

        A_blocks.append(A_phys_blk)
        b_blocks.append(b_2d_blk)

        rnk, smax, smin, ratio, _ = svd_rank(A_scaled, rel_thresh=rel_svd_thresh)
        block_info.append(
            dict(
                block=tag,
                cond_scaled=cond_safe(A_scaled),
                rank_scaled=rnk,
                svd_sigma_min_scaled=smin,
                svd_ratio_scaled=ratio,
                p_dim=A_phys_blk.shape[1],
            )
        )

    for blk in data_blocks:
        _append_block(
            tag=blk["tag"],
            w_blk=blk["w"],
            sig_blk=blk["sigma"],
            s_ref_blk=blk["s_ref"],
            tau_blk=blk["tau"],
            s_1_blk=blk["s_1"],
            s_2_blk=blk["s_2"],
        )

    A_phys = np.vstack(A_blocks)             # (N,8)
    b_2d   = np.concatenate(b_blocks)        # (N,)
    return A_phys, b_2d, block_info


def _fixed_identifiable_from_diags(mu11: complex, mu22: complex) -> np.ndarray:
    """
    Returns identifiable μ vector entries that are FIXED when diagonals are frozen:
      [Re11, Im11, Re22, Im22, Re(mu11*mu22), Im(mu11*mu22), ?, ?]
    """
    mur11, mui11 = mu11.real, mu11.imag
    mur22, mui22 = mu22.real, mu22.imag
    re_d = mur11 * mur22 - mui11 * mui22
    im_d = mur11 * mui22 + mui11 * mur22
    out = np.array([mur11, mui11, mur22, mui22, re_d, im_d, 0.0, 0.0], float)
    return out


def _summary_from_block_info(block_info, data_blocks, rel_svd_thresh):
    def summarize(tag):
        vals = [d for d in block_info if d["block"] == tag]
        if not vals:
            return {}
        ranks = [v["rank_scaled"] for v in vals if np.isfinite(v["rank_scaled"])]
        conds = [v["cond_scaled"] for v in vals if np.isfinite(v["cond_scaled"])]
        ratios = [v["svd_ratio_scaled"] for v in vals if np.isfinite(v["svd_ratio_scaled"])]
        return dict(
            rank_scaled_min=int(np.min(ranks)) if ranks else None,
            rank_scaled_max=int(np.max(ranks)) if ranks else None,
            cond_scaled_min=float(np.min(conds)) if conds else np.nan,
            cond_scaled_max=float(np.max(conds)) if conds else np.nan,
            svd_ratio_scaled_min=float(np.min(ratios)) if ratios else np.nan,
            num_blocks=len(vals),
        )

    info = {}
    unique_tags = sorted(list(set(b["tag"] for b in data_blocks)))
    for tag in unique_tags:
        info[f"{tag}_summary"] = summarize(tag)
    return info


def solve_mu_per_R_freeze_diag_rank1_mag_phasefree(
    config,
    n,
    data_blocks,
    mu11_fixed: complex,
    mu22_fixed: complex,
    quiet=False,
    prev_p_opt=None,
):
    """
    HYBRID strategy: hybrid_freeze_diag_rank1_mag_phasefree

    - μ11, μ22 are FROZEN to provided complex values.
    - Enforce |μ12|=|μ21| with magnitude implied by diagonals:
          a12 = 0.5*(log|μ11| + log|μ22|)
      but phases are free:
          μ12 = exp(a12) * e^{i ph12}
          μ21 = exp(a12) * e^{i ph21}

    Optimization variables: q = [ph12, ph21]  (2 params)
    Returns: p_opt (8,) real parameters [Re11,Im11,Re22,Im22,Re12,Im12,Re21,Im21]
    """
    rel_col_thresh = float(getattr(config, "mu_col_rel_thresh", 1e-3))
    rel_svd_thresh = float(getattr(config, "mu_svd_rel_thresh", 1e-10))

    A_phys, b_2d, block_info = _build_Aphys_and_b(
        config, n, data_blocks,
        rel_col_thresh=rel_col_thresh,
        rel_svd_thresh=rel_svd_thresh,
    )

    # fixed diag parts in identifiable space
    fixed_id = _fixed_identifiable_from_diags(mu11_fixed, mu22_fixed)

    # implied coupling log-magnitude from diagonals
    a11, _ = complex_to_q(mu11_fixed)
    a22, _ = complex_to_q(mu22_fixed)
    a12 = 0.5 * (a11 + a22)

    # bounds
    phi_max_c = float(getattr(config, "mu_cpl_phase_max", np.pi))
    lower = np.array([-phi_max_c, -phi_max_c], float)
    upper = np.array([+phi_max_c, +phi_max_c], float)

    # init
    mu12_prior = getattr(config, "mu12_prior", 1.0 + 0.0j)
    mu21_prior = getattr(config, "mu21_prior", mu12_prior)

    q0 = np.array([
        np.clip(np.angle(mu12_prior), -phi_max_c, +phi_max_c),
        np.clip(np.angle(mu21_prior), -phi_max_c, +phi_max_c),
    ], float)

    # warm-start (coupling phases only)
    if prev_p_opt is not None:
        try:
            pprev = np.asarray(prev_p_opt, float).ravel()
            if pprev.size >= 8:
                mur12, mui12, mur21, mui21 = pprev[4], pprev[5], pprev[6], pprev[7]
                q0[0] = np.clip(np.arctan2(mui12, mur12), lower[0], upper[0])
                q0[1] = np.clip(np.arctan2(mui21, mur21), lower[1], upper[1])
                if not quiet:
                    logger.info("[Hybrid freeze-diag | warm-start] phases from previous R.")
        except Exception as e:
            logger.warning(f"[Hybrid freeze-diag | warm-start] failed: {e}")

    lam_cont = float(getattr(config, "mu_continuation_lambda", 0.0))

    def residual(q):
        ph12, ph21 = float(q[0]), float(q[1])

        mu12 = _cplx_from_a_phi(a12, ph12)
        mu21 = _cplx_from_a_phi(a12, ph21)

        # build full identifiable μ vector
        mur12, mui12 = mu12.real, mu12.imag
        mur21, mui21 = mu21.real, mu21.imag
        re_c = mur12 * mur21 - mui12 * mui21
        im_c = mur12 * mui21 + mui12 * mur21

        mu_id = fixed_id.copy()
        mu_id[6] = re_c
        mu_id[7] = im_c

        r_phys = A_phys @ mu_id - b_2d
        Rv = r_phys / np.sqrt(max(r_phys.size, 1))

        # very light continuation on phases if requested
        if lam_cont > 0 and prev_p_opt is not None:
            try:
                pprev = np.asarray(prev_p_opt, float).ravel()
                if pprev.size >= 8:
                    ph12_prev = np.arctan2(pprev[5], pprev[4])
                    ph21_prev = np.arctan2(pprev[7], pprev[6])
                    reg = np.sqrt(lam_cont) * np.array([ph12 - ph12_prev, ph21 - ph21_prev], float)
                    return np.concatenate([Rv, reg])
            except Exception:
                pass

        bad = ~np.isfinite(Rv)
        Rv[bad] = 1e6
        return Rv

    method = getattr(config, "lsq_method", "trf")
    if method == "trf":
        res = least_squares(residual, q0, method=method, bounds=(lower, upper))
    else:
        res = least_squares(residual, q0, method=method)

    ph12_opt, ph21_opt = res.x
    mu12_opt = _cplx_from_a_phi(a12, ph12_opt)
    mu21_opt = _cplx_from_a_phi(a12, ph21_opt)

    p_opt = np.array([
        mu11_fixed.real, mu11_fixed.imag,
        mu22_fixed.real, mu22_fixed.imag,
        mu12_opt.real,  mu12_opt.imag,
        mu21_opt.real,  mu21_opt.imag,
    ], float)

    # diagnostics
    mu_id = build_mu_from_p_phys_collapsed(p_opt)
    r_phys = A_phys @ mu_id - b_2d
    N = max(r_phys.size, 1)
    phys_cost = 0.5 * float(np.dot(r_phys, r_phys)) / N
    phys_rms  = float(np.linalg.norm(r_phys) / np.sqrt(N))
    mu_dict = _extract_mu_dict_from_p(p_opt)
    info = dict(
        mu_fit_strategy="hybrid_freeze_diag_sym_coupling",
        global_cost=float(res.cost),
        residual_norm=float(np.linalg.norm(residual(res.x))),
        phys_cost=float(phys_cost),
        phys_rms=float(phys_rms),
        cond_A_phys=cond_safe(A_phys),
        rank_A_phys=svd_rank(A_phys, rel_thresh=rel_svd_thresh)[0],
        mu_svd_rel_thresh=rel_svd_thresh,
        p_dim=int(p_opt.size),
        A_phys_shape=A_phys.shape,
        used_warm_start=bool(prev_p_opt is not None),
        mu11=mu_dict["mu11"],
        mu22=mu_dict["mu22"],
        mu12=mu_dict["mu12"],
        mu21=mu_dict["mu21"],
    )
    info.update(_summary_from_block_info(block_info, data_blocks, rel_svd_thresh))

    return p_opt, info


def solve_mu_per_R_freeze_diag_sym_coupling(
    config,
    n,
    data_blocks,
    mu11_fixed: complex,
    mu22_fixed: complex,
    quiet=False,
    prev_p_opt=None,
):
    """
    HYBRID strategy: hybrid_freeze_diag_sym_coupling

    - μ11, μ22 are FROZEN to provided complex values.
    - Enforce μ12 = μ21, but μ12 magnitude+phase are learned.

    Optimization variables: q = [a12, ph12]  (2 params)
    Returns: p_opt (6,) real parameters [Re11,Im11,Re22,Im22,Re12,Im12]
    """
    rel_col_thresh = float(getattr(config, "mu_col_rel_thresh", 1e-3))
    rel_svd_thresh = float(getattr(config, "mu_svd_rel_thresh", 1e-10))

    A_phys, b_2d, block_info = _build_Aphys_and_b(
        config, n, data_blocks,
        rel_col_thresh=rel_col_thresh,
        rel_svd_thresh=rel_svd_thresh,
    )

    # fixed diag parts in identifiable space
    fixed_id = _fixed_identifiable_from_diags(mu11_fixed, mu22_fixed)

    # bounds
    a_min_c = float(getattr(config, "mu_cpl_logmag_min", -2.0))
    a_max_c = float(getattr(config, "mu_cpl_logmag_max", +0.2))
    phi_max_c = float(getattr(config, "mu_cpl_phase_max", np.pi))

    lower = np.array([a_min_c, -phi_max_c], float)
    upper = np.array([a_max_c, +phi_max_c], float)

    # init
    mu12_prior = getattr(config, "mu12_prior", 1.0 + 0.0j)
    a12_0, ph12_0 = complex_to_q(mu12_prior)

    q0 = np.array([
        np.clip(a12_0, a_min_c, a_max_c),
        np.clip(ph12_0, -phi_max_c, +phi_max_c),
    ], float)

    # warm-start from previous R (extract μ12)
    if prev_p_opt is not None:
        try:
            pprev = np.asarray(prev_p_opt, float).ravel()
            if pprev.size >= 6:
                mur12, mui12 = pprev[4], pprev[5]
                a_prev, ph_prev = complex_to_q(mur12 + 1j*mui12)
                q0[0] = np.clip(a_prev, lower[0], upper[0])
                q0[1] = np.clip(ph_prev, lower[1], upper[1])
                if not quiet:
                    logger.info("[Hybrid freeze-diag | warm-start] mu12 from previous R.")
        except Exception as e:
            logger.warning(f"[Hybrid freeze-diag | warm-start] failed: {e}")

    lam_cont = float(getattr(config, "mu_continuation_lambda", 0.0))

    def residual(q):
        a12, ph12 = float(q[0]), float(q[1])
        mu12 = _cplx_from_a_phi(a12, ph12)
        mu21 = mu12

        mur12, mui12 = mu12.real, mu12.imag
        mur21, mui21 = mur12, mui12

        re_c = mur12 * mur21 - mui12 * mui21
        im_c = mur12 * mui21 + mui12 * mur21

        mu_id = fixed_id.copy()
        mu_id[6] = re_c
        mu_id[7] = im_c

        r_phys = A_phys @ mu_id - b_2d
        Rv = r_phys / np.sqrt(max(r_phys.size, 1))

        # optional continuation on [a12,ph12]
        if lam_cont > 0 and prev_p_opt is not None:
            try:
                pprev = np.asarray(prev_p_opt, float).ravel()
                if pprev.size >= 6:
                    a_prev, ph_prev = complex_to_q(pprev[4] + 1j*pprev[5])
                    reg = np.sqrt(lam_cont) * np.array([a12 - a_prev, ph12 - ph_prev], float)
                    return np.concatenate([Rv, reg])
            except Exception:
                pass

        bad = ~np.isfinite(Rv)
        Rv[bad] = 1e6
        return Rv

    method = getattr(config, "lsq_method", "trf")
    if method == "trf":
        res = least_squares(residual, q0, method=method, bounds=(lower, upper))
    else:
        res = least_squares(residual, q0, method=method)

    a12_opt, ph12_opt = res.x
    mu12_opt = _cplx_from_a_phi(a12_opt, ph12_opt)

    p_opt = np.array([
        mu11_fixed.real, mu11_fixed.imag,
        mu22_fixed.real, mu22_fixed.imag,
        mu12_opt.real,  mu12_opt.imag,
    ], float)

    # diagnostics (convert to identifiable using existing helper)
    mu_id = build_mu_from_p_phys_collapsed(p_opt)  # handles p.size==6 (mu21=mu12)
    r_phys = A_phys @ mu_id - b_2d
    N = max(r_phys.size, 1)
    phys_cost = 0.5 * float(np.dot(r_phys, r_phys)) / N
    phys_rms  = float(np.linalg.norm(r_phys) / np.sqrt(N))
    mu_dict = _extract_mu_dict_from_p(p_opt)
    info = dict(
        mu_fit_strategy="hybrid_freeze_diag_sym_coupling",
        global_cost=float(res.cost),
        residual_norm=float(np.linalg.norm(residual(res.x))),
        phys_cost=float(phys_cost),
        phys_rms=float(phys_rms),
        cond_A_phys=cond_safe(A_phys),
        rank_A_phys=svd_rank(A_phys, rel_thresh=rel_svd_thresh)[0],
        mu_svd_rel_thresh=rel_svd_thresh,
        p_dim=int(p_opt.size),
        mu11=mu_dict["mu11"],
        mu22=mu_dict["mu22"],
        mu12=mu_dict["mu12"],
        mu21=mu_dict["mu21"],
        A_phys_shape=A_phys.shape,
        used_warm_start=bool(prev_p_opt is not None),
    )
    info.update(_summary_from_block_info(block_info, data_blocks, rel_svd_thresh))

    return p_opt, info

def solve_mu_per_R_hybrid_analytic_mu12_from_diag(
    config,
    n,
    data_blocks,
    mu11_fixed: complex,
    mu22_fixed: complex,
    quiet=False,
):
    """
    HYBRID strategy: hybrid_analytic_mu12_from_diag

    - μ11 and μ22 are taken as fixed (typically linear fits).
    - μ12 is set analytically to sqrt(μ11*μ22) with Re(μ12) >= 0.
    - μ21 is implied as μ12 (so p is length 6).
    """
    mu12 = _sqrt_with_pos_real(mu11_fixed * mu22_fixed)

    p_opt = np.array(
        [mu11_fixed.real, mu11_fixed.imag, mu22_fixed.real, mu22_fixed.imag, mu12.real, mu12.imag],
        float,
    )

    # compute physics residual for diagnostics
    rel_col_thresh = float(getattr(config, "mu_col_rel_thresh", 1e-3))
    rel_svd_thresh = float(getattr(config, "mu_svd_rel_thresh", 1e-10))
    A_phys, b_2d, _ = _build_Aphys_and_b(config, n, data_blocks, rel_col_thresh, rel_svd_thresh)

    mu_id = build_mu_from_p_phys_collapsed(p_opt)
    r_phys = A_phys @ mu_id - b_2d
    N = max(r_phys.size, 1)
    phys_cost = 0.5 * float(np.dot(r_phys, r_phys)) / N
    phys_rms  = float(np.linalg.norm(r_phys) / np.sqrt(N))
    mu_dict = _extract_mu_dict_from_p(p_opt)
    info = dict(
        mu_fit_strategy="hybrid_analytic_mu12_from_diag",
        global_cost=np.nan,
        residual_norm=float(np.linalg.norm(r_phys) / np.sqrt(N)),
        phys_cost=float(phys_cost),
        phys_rms=float(phys_rms),
        cond_A_phys=cond_safe(A_phys),
        rank_A_phys=svd_rank(A_phys, rel_thresh=rel_svd_thresh)[0],
        mu_svd_rel_thresh=rel_svd_thresh,
        p_dim=int(p_opt.size),
        note="analytic: mu12 = sqrt(mu11*mu22) with Re(mu12)>=0",
        mu11=mu_dict["mu11"],
        mu22=mu_dict["mu22"],
        mu12=mu_dict["mu12"],
        mu21=mu_dict["mu21"],
        A_phys_shape=A_phys.shape,
    )
    return p_opt, info
def solve_mu_per_R_freeze_diag_rank1_mag_antiphase(
    config,
    n,
    data_blocks,
    mu11_fixed: complex,
    mu22_fixed: complex,
    quiet=False,
    prev_p_opt=None,
):
    """
    HYBRID strategy: hybrid_freeze_diag_rank1_mag_antiphase

    - μ11, μ22 are FROZEN to provided complex values.
    - Enforce |μ12|=|μ21| with magnitude implied by diagonals:
          a12 = 0.5*(log|μ11| + log|μ22|)
      and enforce strict anti-phase:
          φ21 = -φ12
      so only ONE phase DOF.

    Optimization variables: q = [φ]  (1 param) where:
        μ12 = exp(a12) * e^{i φ}
        μ21 = exp(a12) * e^{-i φ}

    Returns:
        p_opt (8,) real parameters:
        [Re11,Im11,Re22,Im22,Re12,Im12,Re21,Im21]
    """
    rel_col_thresh = float(getattr(config, "mu_col_rel_thresh", 1e-3))
    rel_svd_thresh = float(getattr(config, "mu_svd_rel_thresh", 1e-10))

    A_phys, b_2d, block_info = _build_Aphys_and_b(
        config, n, data_blocks,
        rel_col_thresh=rel_col_thresh,
        rel_svd_thresh=rel_svd_thresh,
    )

    # fixed diag parts in identifiable space
    fixed_id = _fixed_identifiable_from_diags(mu11_fixed, mu22_fixed)

    # implied coupling log-magnitude from diagonals
    a11, _ = complex_to_q(mu11_fixed)
    a22, _ = complex_to_q(mu22_fixed)
    a12 = 0.5 * (a11 + a22)

    # bounds (single phase)
    phi_max_c = float(getattr(config, "mu_cpl_phase_max", np.pi))
    lower = np.array([-phi_max_c], float)
    upper = np.array([+phi_max_c], float)

    # init (use mu12_prior angle; mu21 implied by anti-phase)
    mu12_prior = getattr(config, "mu12_prior", 1.0 + 0.0j)
    q0 = np.array([np.clip(np.angle(mu12_prior), -phi_max_c, +phi_max_c)], float)

    # warm-start (project previous phases onto anti-phase manifold)
    if prev_p_opt is not None:
        try:
            pprev = np.asarray(prev_p_opt, float).ravel()
            if pprev.size >= 8:
                ph12_prev = np.arctan2(pprev[5], pprev[4])
                ph21_prev = np.arctan2(pprev[7], pprev[6])
                ph_proj = 0.5 * (ph12_prev - ph21_prev)  # best-fit antisym projection
                q0[0] = np.clip(ph_proj, lower[0], upper[0])
                if not quiet:
                    logger.info("[Hybrid freeze-diag antiphase | warm-start] projected phase from previous R.")
        except Exception as e:
            logger.warning(f"[Hybrid freeze-diag antiphase | warm-start] failed: {e}")

    lam_cont = float(getattr(config, "mu_continuation_lambda", 0.0))

    def residual(q):
        ph = float(q[0])
        ph12, ph21 = ph, -ph

        mu12 = _cplx_from_a_phi(a12, ph12)
        mu21 = _cplx_from_a_phi(a12, ph21)

        # coupling product
        mur12, mui12 = mu12.real, mu12.imag
        mur21, mui21 = mu21.real, mu21.imag
        re_c = mur12 * mur21 - mui12 * mui21
        im_c = mur12 * mui21 + mui12 * mur21

        mu_id = fixed_id.copy()
        mu_id[6] = re_c
        mu_id[7] = im_c

        r_phys = A_phys @ mu_id - b_2d
        Rv = r_phys / np.sqrt(max(r_phys.size, 1))

        # optional continuation on the single phase (projected)
        if lam_cont > 0 and prev_p_opt is not None:
            try:
                pprev = np.asarray(prev_p_opt, float).ravel()
                if pprev.size >= 8:
                    ph12_prev = np.arctan2(pprev[5], pprev[4])
                    ph21_prev = np.arctan2(pprev[7], pprev[6])
                    ph_prev_proj = 0.5 * (ph12_prev - ph21_prev)
                    reg = np.sqrt(lam_cont) * np.array([ph - ph_prev_proj], float)
                    out = np.concatenate([Rv, reg])
                else:
                    out = Rv
            except Exception:
                out = Rv
        else:
            out = Rv

        bad = ~np.isfinite(out)
        out[bad] = 1e6
        return out

    method = getattr(config, "lsq_method", "trf")
    if method == "trf":
        res = least_squares(residual, q0, method=method, bounds=(lower, upper))
    else:
        res = least_squares(residual, q0, method=method)

    ph_opt = float(res.x[0])
    mu12_opt = _cplx_from_a_phi(a12,  ph_opt)
    mu21_opt = _cplx_from_a_phi(a12, -ph_opt)

    p_opt = np.array([
        mu11_fixed.real, mu11_fixed.imag,
        mu22_fixed.real, mu22_fixed.imag,
        mu12_opt.real,   mu12_opt.imag,
        mu21_opt.real,   mu21_opt.imag,
    ], float)

    # diagnostics
    mu_id = build_mu_from_p_phys_collapsed(p_opt)
    r_phys = A_phys @ mu_id - b_2d
    N = max(r_phys.size, 1)
    phys_cost = 0.5 * float(np.dot(r_phys, r_phys)) / N
    phys_rms  = float(np.linalg.norm(r_phys) / np.sqrt(N))
    mu_dict = _extract_mu_dict_from_p(p_opt)
    info = dict(
        mu_fit_strategy="hybrid_freeze_diag_rank1_mag_antiphase",
        global_cost=float(res.cost),
        residual_norm=float(np.linalg.norm(residual(res.x))),
        phys_cost=float(phys_cost),
        phys_rms=float(phys_rms),
        cond_A_phys=cond_safe(A_phys),
        rank_A_phys=svd_rank(A_phys, rel_thresh=rel_svd_thresh)[0],
        mu_svd_rel_thresh=rel_svd_thresh,
        p_dim=int(p_opt.size),
        mu11=mu_dict["mu11"],
        mu22=mu_dict["mu22"],
        mu12=mu_dict["mu12"],
        mu21=mu_dict["mu21"],
        A_phys_shape=A_phys.shape,
        used_warm_start=bool(prev_p_opt is not None),
    )
    info.update(_summary_from_block_info(block_info, data_blocks, rel_svd_thresh))

    return p_opt, info
def solve_mu_per_R_dispatch(
    config,
    n,
    data_blocks,
    quiet=False,
    init_mu11: complex | None = None,
    init_mu22: complex | None = None,
    prev_p_opt=None,
):
    """
    Single entry point for ALL second-order μ strategies (base + hybrid).
    Controlled by config.mu_strategy (preferred) or config.mu_fit_strategy (fallback).

    Returns: p_opt, info
    """
    strategy = resolve_mu_strategy(config)

    # --- HYBRID ---
    if strategy == "hybrid_analytic_mu12_from_diag":
        if init_mu11 is None or init_mu22 is None:
            raise ValueError(f"{strategy} requires init_mu11 and init_mu22 (linear fits).")
        return solve_mu_per_R_hybrid_analytic_mu12_from_diag(
            config=config,
            n=n,
            data_blocks=data_blocks,
            mu11_fixed=init_mu11,
            mu22_fixed=init_mu22,
            quiet=quiet,
        )

    if strategy == "hybrid_freeze_diag_rank1_mag_phasefree":
        if init_mu11 is None or init_mu22 is None:
            raise ValueError(f"{strategy} requires init_mu11 and init_mu22 (linear fits).")
        return solve_mu_per_R_freeze_diag_rank1_mag_phasefree(
            config=config,
            n=n,
            data_blocks=data_blocks,
            mu11_fixed=init_mu11,
            mu22_fixed=init_mu22,
            quiet=quiet,
            prev_p_opt=prev_p_opt,
        )

    if strategy == "hybrid_freeze_diag_sym_coupling":
        if init_mu11 is None or init_mu22 is None:
            raise ValueError(f"{strategy} requires init_mu11 and init_mu22 (linear fits).")
        return solve_mu_per_R_freeze_diag_sym_coupling(
            config=config,
            n=n,
            data_blocks=data_blocks,
            mu11_fixed=init_mu11,
            mu22_fixed=init_mu22,
            quiet=quiet,
            prev_p_opt=prev_p_opt,
        )
    
    if strategy == "hybrid_freeze_diag_rank1_mag_antiphase":
        return solve_mu_per_R_freeze_diag_rank1_mag_antiphase(
            config=config,
            n=n,
            data_blocks=data_blocks,
            mu11_fixed=init_mu11,
            mu22_fixed=init_mu22,
            quiet=quiet,
            prev_p_opt=prev_p_opt,
        )

    # --- BASE ---
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown mu_strategy='{strategy}'. "
            f"Base: {list(STRATEGIES.keys())}. "
            f"Hybrid: {sorted(HYBRID_STRATEGIES)}."
        )

    config.mu_fit_strategy = strategy

    return solve_mu_per_R(
        config=config,
        n=n,
        data_blocks=data_blocks,
        quiet=quiet,
        init_mu11=init_mu11,
        init_mu22=init_mu22,
        prev_p_opt=prev_p_opt,
    )


# functions may be added later if needed

# def build_n_weights(n, weights=None, *, kind="exp", beta=2.0, power=1.0, eps=1e-6):
#     """
#     Returns weights for samples corresponding to n[1:].
#     Output shape: (m,) where m = len(n)-1.
#     Higher weights for earlier/smaller n.

#     weights can be:
#       - None: use rule (kind,beta/power)
#       - scalar: constant weight
#       - array-like of shape (m,)
#       - callable: weights = f(n_mid) -> (m,)
#     """
#     n = np.asarray(n).ravel()
#     m = n.size - 1
#     n_mid = n[1:]

#     if weights is None:
#         if kind == "exp":
#             # exp weighting in n-value (stable)
#             x = n_mid - float(n_mid.min())
#             w = np.exp(-beta * x / (float(x.max()) + eps if x.max() > 0 else 1.0))
#         elif kind == "power":
#             w = 1.0 / np.power(n_mid + eps, power)
#         elif kind == "index_exp":
#             k = np.arange(m)
#             w = np.exp(-beta * k)
#         else:
#             raise ValueError(f"Unknown weight kind: {kind}")
#         return w.astype(float)

#     if np.isscalar(weights):
#         return np.full(m, float(weights), float)

#     if callable(weights):
#         w = np.asarray(weights(n_mid), float).ravel()
#         if w.size != m:
#             raise ValueError(f"Callable weights must return shape ({m},), got {w.shape}")
#         return w

#     w = np.asarray(weights, float).ravel()
#     if w.size != m:
#         raise ValueError(f"weights must have shape ({m},), got {w.shape}")
#     return w
