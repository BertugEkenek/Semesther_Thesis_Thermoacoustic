# fitting/second_order_fit.py

import numpy as np
from scipy.optimize import least_squares
from utils import logger

# ============================================================
#          SECOND-ORDER μ-FIT — CLEAN GLOBAL PER-R SOLVER
# ============================================================

# -----------------------------
# numerics helpers
# -----------------------------
def cond_safe(A) -> float:
    try:
        return float(np.linalg.cond(A))
    except Exception:
        return float(np.inf)
    
def build_n_weights(n, weights=None, *, kind="exp", beta=2.0, power=1.0, eps=1e-6):
    """
    Returns weights for samples corresponding to n[1:].
    Output shape: (m,) where m = len(n)-1.
    Higher weights for earlier/smaller n.

    weights can be:
      - None: use rule (kind,beta/power)
      - scalar: constant weight
      - array-like of shape (m,)
      - callable: weights = f(n_mid) -> (m,)
    """
    n = np.asarray(n).ravel()
    m = n.size - 1
    n_mid = n[1:]

    if weights is None:
        if kind == "exp":
            # exp weighting in n-value (stable)
            x = n_mid - float(n_mid.min())
            w = np.exp(-beta * x / (float(x.max()) + eps if x.max() > 0 else 1.0))
        elif kind == "power":
            w = 1.0 / np.power(n_mid + eps, power)
        elif kind == "index_exp":
            k = np.arange(m)
            w = np.exp(-beta * k)
        else:
            raise ValueError(f"Unknown weight kind: {kind}")
        return w.astype(float)

    if np.isscalar(weights):
        return np.full(m, float(weights), float)

    if callable(weights):
        w = np.asarray(weights(n_mid), float).ravel()
        if w.size != m:
            raise ValueError(f"Callable weights must return shape ({m},), got {w.shape}")
        return w

    w = np.asarray(weights, float).ravel()
    if w.size != m:
        raise ValueError(f"weights must have shape ({m},), got {w.shape}")
    return w

def p_to_q_from_previous(p_prev, enforce_symmetry, hard=False, bake=False):
    """
    Convert solved p-vector from previous R into q-space initialization.
    Returns q0 array.
    """
    if hard or bake:
        # only diagonals exist
        mur11, mui11, mur22, mui22 = p_prev[:4]
        a11, ph11 = complex_to_q(mur11 + 1j * mui11)
        a22, ph22 = complex_to_q(mur22 + 1j * mui22)
        return np.array([a11, ph11, a22, ph22], float)

    if enforce_symmetry:
        mur11, mui11, mur22, mui22, mur12, mui12 = p_prev
        a11, ph11 = complex_to_q(mur11 + 1j * mui11)
        a22, ph22 = complex_to_q(mur22 + 1j * mui22)
        a12, ph12 = complex_to_q(mur12 + 1j * mui12)
        return np.array([a11, ph11, a22, ph22, a12, ph12], float)

    # full (8 dof)
    mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21 = p_prev
    a11, ph11 = complex_to_q(mur11 + 1j * mui11)
    a22, ph22 = complex_to_q(mur22 + 1j * mui22)
    a12, ph12 = complex_to_q(mur12 + 1j * mui12)
    a21, ph21 = complex_to_q(mur21 + 1j * mui21)
    return np.array([a11, ph11, a22, ph22, a12, ph12, a21, ph21], float)

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


def ensure_2d(arr):
    arr = np.asarray(arr)
    return arr[:, None] if arr.ndim == 1 else arr


def complex_to_q(mu: complex):
    """Map complex μ to (log|μ|, arg(μ))."""
    r = max(abs(mu), 1e-12)
    return np.log(r), np.arctan2(mu.imag, mu.real)


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
    
    m = len(n) - 1
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
    M_all[:, 1, 11] = 2*C12*C21*(Cin*Crn*c**2 - Cin*Crn*s**2 +
                                 Cin**2*c*s - Crn**2*c*s)

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
    Collapse 12 → 8 identifiable groups, optionally bake → 6.
    Returns:
      A_phys: (N, p_dim_phys)  where p_dim_phys is 8 or 6
      b_2d:   (N,)
      meta:   dict
    """
    pdim_basis = M_all.shape[2]
    M_2d = M_all.reshape(-1, pdim_basis)

    # 12 → 8 (STRUCTURAL)
    m = M_all.shape[0]
    M8 = np.zeros((m, 2, 8), float)
    M8[:, :, 0:4] = M_all[:, :, 0:4]
    M8[:, :, 4] = 0.5 * (M_all[:, :, 4] - M_all[:, :, 7])   # Re(mu11*mu22)
    M8[:, :, 5] = 0.5 * (M_all[:, :, 5] + M_all[:, :, 6])   # Im(mu11*mu22)
    M8[:, :, 6] = 0.5 * (M_all[:, :, 8] - M_all[:, :, 11])  # Re(mu12*mu21)
    M8[:, :, 7] = 0.5 * (M_all[:, :, 9] + M_all[:, :, 10])  # Im(mu12*mu21)

    bake = bool(getattr(config, "mu_bake_rank_one", False))
    if bake:
        # 8 → 6: merge diagonal-product and coupling-product blocks
        M6 = np.zeros((m, 2, 6), float)
        M6[:, :, 0:4] = M8[:, :, 0:4]
        M6[:, :, 4:6] = M8[:, :, 4:6] + M8[:, :, 6:8]
        A_phys = M6.reshape(-1, 6)
        return A_phys, bake

    A_phys = M8.reshape(-1, 8)
    return A_phys, bake

# ============================================================
#  q ↔ μ ↔ p mapping
# ============================================================
def q_to_mu_complex(q, enforce_symmetry, hard_constraint=False, sign=+1, bake=False):
    """
    q parametrization uses (a,phi) with μ = exp(a) * exp(i phi).
    - hard_constraint: only diagonals are free; coupling is implicit (rank-one).
      coupling is constructed branch-stable (avoid complex sqrt branch cuts).
    - bake: same rank-one idea but without forcing sign selection via two solves.
    """
    if hard_constraint:
        a11, phi11, a22, phi22 = q
        mu11 = np.exp(a11) * (np.cos(phi11) + 1j*np.sin(phi11))
        mu22 = np.exp(a22) * (np.cos(phi22) + 1j*np.sin(phi22))

        a12 = 0.5 * (a11 + a22)
        phi12 = 0.5 * (phi11 + phi22)
        mu12_base = np.exp(a12) * (np.cos(phi12) + 1j*np.sin(phi12))

        mu12 = sign * mu12_base
        mu21 = mu12
        return mu11, mu22, mu12, mu21

    if bake:
        # only diagonals in q; coupling is implied
        a11, phi11, a22, phi22 = q
        mu11 = np.exp(a11) * (np.cos(phi11) + 1j*np.sin(phi11))
        mu22 = np.exp(a22) * (np.cos(phi22) + 1j*np.sin(phi22))
        a12 = 0.5 * (a11 + a22)
        phi12 = 0.5 * (phi11 + phi22)
        mu12 = sign * np.exp(a12) * (np.cos(phi12) + 1j*np.sin(phi12))
        mu21 = mu12
        return mu11, mu22, mu12, mu21

    # full soft model
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


def q_to_p(q, enforce_symmetry, hard_constraint=False, sign=+1, bake=False):
    """
    p is the "physical parameter vector" returned by the solver:
      symmetric: [Re11, Im11, Re22, Im22, Re12, Im12]
      full:      [Re11, Im11, Re22, Im22, Re12, Im12, Re21, Im21]
    For bake: still returns the same p-layout (so upstream stays consistent).
    """
    mu11, mu22, mu12, mu21 = q_to_mu_complex(
        q, enforce_symmetry, hard_constraint=hard_constraint, sign=sign, bake=bake
    )

    if enforce_symmetry:
        return np.array([mu11.real, mu11.imag, mu22.real, mu22.imag, mu12.real, mu12.imag], float)

    return np.array([mu11.real, mu11.imag, mu22.real, mu22.imag,
                     mu12.real, mu12.imag, mu21.real, mu21.imag], float)


def build_mu_from_p_phys_collapsed(p, enforce_symmetry):
    """
    Build identifiable μ-vector of length 8 used by A_phys (non-baked):
      [Re11, Im11, Re22, Im22, Re(mu11*mu22), Im(mu11*mu22), Re(mu12*mu21), Im(mu12*mu21)]
    """
    if enforce_symmetry:
        mur11, mui11, mur22, mui22, mur12, mui12 = p
        mur21, mui21 = mur12, mui12
    else:
        mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21 = p

    # products:
    re_d = mur11*mur22 - mui11*mui22
    im_d = mur11*mui22 + mui11*mur22

    re_c = mur12*mur21 - mui12*mui21
    im_c = mur12*mui21 + mui12*mur21

    return np.array([mur11, mui11, mur22, mui22, re_d, im_d, re_c, im_c], float)


def build_mu_from_p_phys_baked6(p, enforce_symmetry):
    mu8 = build_mu_from_p_phys_collapsed(p, enforce_symmetry)
    return np.array([mu8[0], mu8[1], mu8[2], mu8[3], mu8[4], mu8[5]], float)


# ============================================================
#  solve per R (global stacked nonlinear LSQ)
# ============================================================
def solve_mu_per_R(
    config,
    n,
    data_blocks,
    enforce_symmetry,
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
        'w': array (m, T),
        'sigma': array (m, T),
        's_ref': complex,
        's_1': complex,
        's_2': complex
      }
    """

    hard = bool(getattr(config, "mu_hard_constraint", False))
    bake = bool(getattr(config, "mu_bake_rank_one", False))

    if hard and bake:
        raise ValueError("mu_hard_constraint and mu_bake_rank_one cannot both be True.")

    # -----------------------------
    # build stacked physics system
    # -----------------------------
    A_blocks = []
    b_blocks = []
    block_info = []

    rel_col_thresh = float(getattr(config, "mu_col_rel_thresh", 1e-3))
    rel_svd_thresh = float(getattr(config, "mu_svd_rel_thresh", 1e-10))

    def _append_block(tag, w_blk, sig_blk, s_ref_blk, tau_blk, s_1_blk, s_2_blk):
        m, T = w_blk.shape
        for j in range(T):
            w = w_blk[:, j]
            sig = sig_blk[:, j]

            M_all, b_all = _build_M_all_and_b_all(
                config, tau_blk, w, sig, s_ref_blk, s_1_blk, s_2_blk, n
            )
            A_phys, _ = _collapse_12_to_identifiable(M_all, config)
            b_2d = b_all.reshape(-1)

            col_norms = np.linalg.norm(A_phys, axis=0)
            max_norm = np.max(col_norms) if np.any(col_norms > 0) else 1.0
            eff = np.maximum(col_norms, rel_col_thresh * max_norm)
            A_scaled = A_phys / eff

            A_blocks.append((A_scaled, eff, A_phys))
            b_blocks.append(b_2d)

            rnk, smax, smin, ratio, _ = svd_rank(A_scaled, rel_thresh=rel_svd_thresh)
            block_info.append(dict(
                block=tag,
                time_index=j,
                cond_scaled=cond_safe(A_scaled),
                rank_scaled=rnk,
                svd_sigma_min_scaled=smin,
                svd_ratio_scaled=ratio,
                p_dim=A_phys.shape[1],
            ))

    # Process all blocks
    for blk in data_blocks:
        w_b = ensure_2d(blk['w'])
        s_b = ensure_2d(blk['sigma'])
        
        _append_block(
            tag=blk['tag'],
            w_blk=w_b,
            sig_blk=s_b,
            s_ref_blk=blk['s_ref'],
            tau_blk=blk['tau'],
            s_1_blk=blk['s_1'],
            s_2_blk=blk['s_2']
        )

    A_phys = np.vstack([x[2] for x in A_blocks])
    b_2d = np.concatenate(b_blocks)

    # -----------------------------
    # q-dimension + bounds + init
    # -----------------------------
    if hard or bake:
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

    lower = np.full(q_dim, -np.inf)
    upper = np.full(q_dim, +np.inf)

    lower[:4:2] = a_min
    upper[:4:2] = a_max
    lower[1:4:2] = -phi_max
    upper[1:4:2] = +phi_max

    # -------------------------------------------------
    # linear μ initialization (diagonals)
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
    # WARM START from previous R
    # -------------------------------------------------
    if prev_p_opt is not None:
        try:
            q_prev = p_to_q_from_previous(
                prev_p_opt,
                enforce_symmetry,
                hard=hard,
                bake=bake,
            )

            k = min(q_prev.size, q0.size)
            q0[:k] = np.clip(q_prev[:k], lower[:k], upper[:k])

            if not quiet:
                logger.info("[Second-order μ | warm-start] Using previous R solution.")

        except Exception as e:
            logger.warning(
                f"[Second-order μ | warm-start] Failed, fallback to default init. Reason: {e}"
            )

    # -------------------------------------------------
    # coupling initialization (only if soft + non-baked)
    # -------------------------------------------------
    mu12_prior = getattr(config, "mu12_prior", 1.0 + 0.0j)
    mu21_prior = getattr(config, "mu21_prior", mu12_prior)

    if (not hard) and (not bake):
        a12, ph12 = complex_to_q(mu12_prior)
        q0[4:6] = [
            np.clip(a12, a_min_c, a_max_c),
            np.clip(ph12, -phi_max_c, phi_max_c),
        ]
        lower[4:6] = [a_min_c, -phi_max_c]
        upper[4:6] = [a_max_c, +phi_max_c]

        if not enforce_symmetry:
            a21, ph21 = complex_to_q(mu21_prior)
            q0[6:8] = [
                np.clip(a21, a_min_c, a_max_c),
                np.clip(ph21, -phi_max_c, phi_max_c),
            ]
            lower[6:8] = [a_min_c, -phi_max_c]
            upper[6:8] = [a_max_c, +phi_max_c]

    p_init = None
    p_cont = None
    if prev_p_opt is not None:
        p_cont = np.asarray(prev_p_opt, float)
    if (init_mu11 is not None) or (init_mu22 is not None):
        mu11_0 = init_mu11 if init_mu11 is not None else (1.0 + 0.0j)
        mu22_0 = init_mu22 if init_mu22 is not None else (1.0 + 0.0j)

        if enforce_symmetry:
            p_init = np.array([mu11_0.real, mu11_0.imag, mu22_0.real, mu22_0.imag,
                               mu12_prior.real, mu12_prior.imag], float)
        else:
            p_init = np.array([mu11_0.real, mu11_0.imag, mu22_0.real, mu22_0.imag,
                               mu12_prior.real, mu12_prior.imag, mu21_prior.real, mu21_prior.imag], float)

        if hard or bake:
            if enforce_symmetry:
                p_init[4:6] = 0.0
            else:
                p_init[4:8] = 0.0

    # if prev_p_opt is not None:
    #     p_init = None

    lam_target = float(getattr(config, "mu_modelIII_lambda", 0.0))
    lam_init   = float(getattr(config, "mu_init_lambda", 0.0))
    lam_one = float(getattr(config, "mu_one_target_lambda", 0.0))
    lam_cont = float(getattr(config, "mu_continuation_lambda", 0.0))

    # target p-vector
    p_target = np.array([1, 0, 1, 0, 1, 0], float) if enforce_symmetry else np.array([1, 0, 1, 0, 1, 0, 1, 0], float)

    # -----------------------------
    # solver for sign (hard/bake) or single run (soft)
    # -----------------------------
    def solve_for_sign(sign):
        def residual(q):
            p = q_to_p(q, enforce_symmetry, hard_constraint=hard, sign=sign, bake=bake)

            # physics mapping
            if bake:
                mu_vec = build_mu_from_p_phys_baked6(p, enforce_symmetry)   # (6,)
            else:
                mu_vec = build_mu_from_p_phys_collapsed(p, enforce_symmetry)  # (8,)

            r_phys = A_phys @ mu_vec - b_2d
            Rv = r_phys / np.sqrt(max(r_phys.size, 1))

            reg_list = []

            if lam_target > 0:
                reg_list.append(np.sqrt(lam_target) * (p - p_target))

            if lam_init > 0 and (p_init is not None):
                reg_list.append(np.sqrt(lam_init) * (p - p_init))
            
            if lam_cont > 0 and (p_cont is not None):
                reg_list.append(
                    np.sqrt(lam_cont) * (p - p_cont)
                )
            if lam_one > 0:
                mur11, _, mur22, _, _, _ = p
                reg_real = np.sqrt(lam_one) * np.array(
                    [mur11 - 1.0, mur22 - 1.0], float
                )
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
        p_opt = q_to_p(q_opt, enforce_symmetry, hard_constraint=hard, sign=sign, bake=bake)

        r = residual(q_opt)
        r = r[np.isfinite(r)]
        rn = float(np.linalg.norm(r)) if r.size else np.nan
        return p_opt, float(res.cost), rn

    if hard or bake:
        p_p, cost_p, rn_p = solve_for_sign(+1)
        p_m, cost_m, rn_m = solve_for_sign(-1)

        if cost_p <= cost_m:
            p_opt, best_cost, best_rn, best_sign = p_p, cost_p, rn_p, +1
        else:
            p_opt, best_cost, best_rn, best_sign = p_m, cost_m, rn_m, -1
    else:
        p_opt, best_cost, best_rn = solve_for_sign(+1)
        best_sign = +1
    
    if bake:
        mu_vec = build_mu_from_p_phys_baked6(p_opt, enforce_symmetry)
    else:
        mu_vec = build_mu_from_p_phys_collapsed(p_opt, enforce_symmetry)

    r_phys = A_phys @ mu_vec - b_2d
    N_phys = max(r_phys.size, 1)

    phys_cost = 0.5 * float(np.dot(r_phys, r_phys)) / N_phys
    phys_rms  = float(np.linalg.norm(r_phys) / np.sqrt(N_phys))


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

    info = dict(
        global_cost=float(best_cost),
        residual_norm=float(best_rn),

        phys_cost=float(phys_cost),
        phys_rms=float(phys_rms),

        cond_A_phys=cond_safe(A_phys),
        rank_A_phys=svd_rank(A_phys, rel_thresh=rel_svd_thresh)[0],
        mu_svd_rel_thresh=rel_svd_thresh,
        p_dim=int(6 if enforce_symmetry else 8),
        mu12_sign=int(best_sign),
        bake_rank_one=bool(bake),
        hard_constraint=bool(hard),
    )

    # Summarize all blocks
    unique_tags = sorted(list(set(b['tag'] for b in data_blocks)))
    for tag in unique_tags:
        info[f"{tag}_summary"] = summarize(tag)

    if (not quiet):
        info["block_info"] = block_info

    return p_opt, info
