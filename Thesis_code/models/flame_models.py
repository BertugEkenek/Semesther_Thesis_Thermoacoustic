import numpy as np
from scipy.special import factorial
from scipy.interpolate import pade

# --- Global caches ---
_pade_cache = {}
_taylor_cache = {}


def _get_pade_polynomials(order: int):
    """
    Return cached [order/order] Padé polynomials for exp(-z).
    """
    if order in _pade_cache:
        return _pade_cache[order]

    coeffs = [(-1)**k / factorial(k) for k in range(2 * order + 1)]
    num_poly, den_poly = pade(coeffs, order)
    _pade_cache[order] = (num_poly, den_poly)
    return num_poly, den_poly


def _get_taylor_polynomial(order: int, tau: float):
    """
    Return cached Taylor polynomial of exp(-tau*s) as a poly1d in s.
    """
    key = (order, tau)
    if key in _taylor_cache:
        return _taylor_cache[key]

    coeffs = [((-tau)**k / factorial(k)) for k in reversed(range(order + 1))]
    poly = np.poly1d(coeffs)
    _taylor_cache[key] = poly
    return poly


def F_taylor(s, n, tau, order=20):
    """
    Compute n * exp(-tau * s) via Taylor series with caching.
    """
    poly = _get_taylor_polynomial(order, tau)
    return n * poly(s)


def F_pade(s, n, tau, order=4):
    """
    Compute a Padé approximation of n * exp(-tau * s) with caching.

    Returns (num, den) exactly as before, so the rest of the code
    (characteristic_poly_model3_*, solve_roots, etc.) stays unchanged.
    """
    num_poly, den_poly = _get_pade_polynomials(order)
    z = tau * s

    # Evaluate at z = tau*s (works for scalar s or poly1d s)
    num = n * np.poly1d(num_poly.coeffs)(z)
    den = np.poly1d(den_poly.coeffs)(z)

    return num, den
