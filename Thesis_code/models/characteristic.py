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

def characteristic_poly_model3_1mode(s, F, n, tau, order, config, mu = 1.0):

    K = config.K
    w1 = config.w[0]
    nu = config.nu
    alpha = config.alpha
    Lambda = config.Lambda

    Fs_numerator, Fs_denominator = F(s, n, tau, order)
    M11 = (s**2 - w1**2) * Fs_denominator - s**2 * Lambda[0] * alpha[0] * nu[0] * K * Fs_numerator * mu
    return M11

def characteristic_poly_model3_2mode(s: complex, F: callable, n: float, tau: float, order: int, config: object, mu: complex, enforce_symmetry: bool):
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

    if enforce_symmetry:
        mu11 = mu[0] + mu[1] * 1j
        mu22 = mu[2] + mu[3] * 1j
        mu12 = mu[4] + mu[5] * 1j 
        mu21 = mu[4] + mu[5] * 1j
        mu11 = complex(np.squeeze(mu11))
        mu22 = complex(np.squeeze(mu22))
        mu12 = complex(np.squeeze(mu12))
        mu21 = complex(np.squeeze(mu21))
    else:
        mu11 = mu[0] + mu[1] * 1j
        mu22 = mu[2] + mu[3] * 1j
        mu12 = mu[4] + mu[5] * 1j 
        mu21 = mu[6] + mu[7] * 1j
        mu11 = complex(np.squeeze(mu11))
        mu22 = complex(np.squeeze(mu22))
        mu12 = complex(np.squeeze(mu12))
        mu21 = complex(np.squeeze(mu21))

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