# pipelines/io_pipeline.py

import os
import numpy as np


def save_reference_and_noisy_solutions(
    solutions: np.ndarray,
    n_values: np.ndarray,
    noise_sigma: float,
    results_dir: str = "./Results/Solutions/",
):
    """
    Save reference and noisy eigenvalues and associated n-values to TXT files.

    Parameters
    ----------
    solutions : np.ndarray
        1D array of complex acoustic eigenvalues (one per n).
    n_values : np.ndarray
        1D array of n values.
    noise_sigma : float
        Standard deviation for Gaussian noise on real & imag parts.
    results_dir : str
        Directory for result .txt files.
    """

    os.makedirs(results_dir, exist_ok=True)

    ref_path = os.path.join(results_dir, "Reference_case.txt")
    ref_n_path = os.path.join(results_dir, "Reference_case_n_values.txt")

    noisy_path = os.path.join(results_dir, "Case_with_noise.txt")
    noisy_n_path = os.path.join(results_dir, "Case_with_noise_n_values.txt")

    # 1) Save noise–free reference acoustic-only eigenvalues
    np.savetxt(ref_path, solutions)

    # ALWAYS save n_values without any noise
    np.savetxt(ref_n_path, n_values)

    # 2) Create noisy version (noise ONLY on eigenvalues!)
    noise_real = np.random.normal(0, noise_sigma, solutions.shape[0])
    noise_imag = np.random.normal(0, noise_sigma, solutions.shape[0])
    noisy_solutions = solutions + noise_real + 1j * noise_imag

    np.savetxt(noisy_path, noisy_solutions)
    np.savetxt(noisy_n_path, n_values)

    print("[TXT SAVE] Saved reference and noisy eigenvalues + n_values")
