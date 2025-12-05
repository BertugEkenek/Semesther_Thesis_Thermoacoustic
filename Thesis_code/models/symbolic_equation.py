import sympy as sp
from utils import logger

def symbolic_equation(coeffs,R):
    R = sp.symbols("R")
    for i in range(2):
        for j in range(2):
            coeffs_ij = coeffs[i, j]
            expr = sum(sp.N(coeffs_ij[k], 6) * R**k for k in range(len(coeffs_ij)))
            logger.info(f"Mu[{i+1}{j+1}](R) = {sp.expand(expr)}\n")
