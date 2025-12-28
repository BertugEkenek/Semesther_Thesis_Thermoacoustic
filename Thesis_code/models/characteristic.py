import numpy as np

def characteristic_poly_model3(s: complex, F: callable, n: float, tau: float, order: int, config: object):
    """
    Parameters
    ----------
    s : complex
        Complex variable
    F : callable
        Flame model. Should take three arguments: s, n, tau, and order.
    n : float
        Non-dimensional frequency
    tau : float
        Time delay
    order : int
        Order of the flame model

    Returns
    -------
    char_poly : complex
        Characteristic polynomial
    """

    K = config.K
    w1 = config.w[0]
    w2 = config.w[1]
    nu = config.nu
    alpha = config.alpha
    Lambda = config.Lambda

    if F.__name__ == 'F_pade':
        Fs_numerator, Fs_denominator = F(s, n, tau, order)
        M11 = (s**2 - w1**2) * Fs_denominator - s**2 * Lambda[0] * alpha[0] * nu[0] * K * Fs_numerator
        M22 = (s**2 - w2**2) * Fs_denominator - s**2 * Lambda[1] * alpha[1] * nu[1] * K * Fs_numerator
        M12 = -s**2 * Lambda[1] * alpha[1] * nu[2] * K * Fs_numerator
        M21 = -s**2 * Lambda[0] * alpha[0] * nu[3] * K * Fs_numerator
        return M11 * M22 - M12 * M21
    else:
        Fs = F(s, n, tau, order)
        M11 = s**2 - w1**2 - s**2 * Lambda[0] * alpha[0] * nu[0] * K * Fs
        M22 = s**2 - w2**2 - s**2 * Lambda[1] * alpha[1] * nu[1] * K * Fs
        M12 = -s**2 * Lambda[1] * alpha[1] * nu[2] * K * Fs
        M21 = -s**2 * Lambda[0] * alpha[0] * nu[3] * K * Fs
        return M11 * M22 - M12 * M21

def characteristic_poly_model3_1mode(
    s, F, n, tau, order, config, mu=1.0, branch_id=1
):
    """
    Branch-aware single-mode characteristic polynomial.

    branch_id:
        1 → first acoustic branch
        2 → second acoustic branch
    """

    if branch_id not in (1, 2):
        raise ValueError(f"branch_id must be 1 or 2, got {branch_id}")

    b = branch_id - 1

    K = config.K
    w = config.w[b]
    Lambda = config.Lambda[b]
    alpha = config.alpha[b]

    # Diagonal nu only
    if b == 0:
        nu_diag = config.nu[0]   # nu11
    else:
        nu_diag = config.nu[1]   # nu22

    Fs_numerator, Fs_denominator = F(s, n, tau, order)

    Mii = (
        (s**2 - w**2) * Fs_denominator
        - s**2 * Lambda * alpha * nu_diag * K * Fs_numerator * mu
    )

    return Mii


def characteristic_poly_model3_2mode(
    s: complex, F: callable, n: float, tau: float, order: int,
    config: object, mu, enforce_symmetry: bool
):
    K = config.K
    w1 = config.w[0]
    w2 = config.w[1]
    nu = config.nu
    alpha = config.alpha
    Lambda = config.Lambda

    mu11, mu22, mu12, mu21 = _parse_mu_second_order(mu, enforce_symmetry=enforce_symmetry)

    if F.__name__ == 'F_pade':
        Fs_numerator, Fs_denominator = F(s, n, tau, order)

        M11 = (s**2 - w1**2) * Fs_denominator - mu11 * s**2 * Lambda[0] * alpha[0] * nu[0] * K * Fs_numerator
        M22 = (s**2 - w2**2) * Fs_denominator - mu22 * s**2 * Lambda[1] * alpha[1] * nu[1] * K * Fs_numerator
        M12 = - mu12 * s**2 * Lambda[1] * alpha[1] * nu[2] * K * Fs_numerator
        M21 = - mu21 * s**2 * Lambda[0] * alpha[0] * nu[3] * K * Fs_numerator

        return M11 * M22 - M12 * M21

    else:
        Fs = F(s, n, tau, order)

        M11 = s**2 - w1**2 - mu11 * s**2 * Lambda[0] * alpha[0] * nu[0] * K * Fs
        M22 = s**2 - w2**2 - mu22 * s**2 * Lambda[1] * alpha[1] * nu[1] * K * Fs
        M12 = - mu12 * s**2 * Lambda[1] * alpha[1] * nu[2] * K * Fs
        M21 = - mu21 * s**2 * Lambda[0] * alpha[0] * nu[3] * K * Fs

        return M11 * M22 - M12 * M21

    

def _is_scalar_mu(mu) -> bool:
    return np.isscalar(mu) or isinstance(mu, (complex, float, int, np.number))

def _mu_to_complex_scalar(x) -> complex:
    return complex(np.asarray(x).squeeze())

def _parse_mu_second_order(mu, enforce_symmetry: bool):
    """
    Accepts multiple mu formats and returns (mu11, mu22, mu12, mu21) as complex scalars.

    Supported:
      - scalar mu: treated as uniform scaling (mu11=mu22=mu12=mu21=mu)
      - symmetric vector (len=6): [Re11,Im11, Re22,Im22, Re12,Im12]
      - full vector (len=8):      [Re11,Im11, Re22,Im22, Re12,Im12, Re21,Im21]
      - baked symmetric (len=4):  [Re11,Im11, Re22,Im22]  with mu12=mu21=sqrt(mu11*mu22)
    """
    if mu is None:
        mu_c = 1.0 + 0.0j
        return mu_c, mu_c, mu_c, mu_c

    # scalar mu case (linear / no correction / legacy)
    if _is_scalar_mu(mu):
        mu_c = complex(mu)
        return mu_c, mu_c, mu_c, mu_c

    mu_arr = np.asarray(mu).squeeze()
    if mu_arr.ndim != 1:
        mu_arr = mu_arr.ravel()

    L = mu_arr.size

    # baked symmetric: [mu11, mu22] only
    if L == 4:
        mu11 = _mu_to_complex_scalar(mu_arr[0] + 1j * mu_arr[1])
        mu22 = _mu_to_complex_scalar(mu_arr[2] + 1j * mu_arr[3])
        # baked constraint: mu12^2 = mu11*mu22  (principal branch)
        mu12 = complex(np.sqrt(mu11 * mu22))
        mu21 = mu12
        return mu11, mu22, mu12, mu21

    if enforce_symmetry:
        if L != 6:
            raise ValueError(
                f"Expected symmetric mu length 6 (or baked length 4), got {L}."
            )
        mu11 = _mu_to_complex_scalar(mu_arr[0] + 1j * mu_arr[1])
        mu22 = _mu_to_complex_scalar(mu_arr[2] + 1j * mu_arr[3])
        mu12 = _mu_to_complex_scalar(mu_arr[4] + 1j * mu_arr[5])
        mu21 = mu12
        return mu11, mu22, mu12, mu21

    # full case
    if L != 8:
        raise ValueError(f"Expected full mu length 8, got {L}.")
    mu11 = _mu_to_complex_scalar(mu_arr[0] + 1j * mu_arr[1])
    mu22 = _mu_to_complex_scalar(mu_arr[2] + 1j * mu_arr[3])
    mu12 = _mu_to_complex_scalar(mu_arr[4] + 1j * mu_arr[5])
    mu21 = _mu_to_complex_scalar(mu_arr[6] + 1j * mu_arr[7])
    return mu11, mu22, mu12, mu21
