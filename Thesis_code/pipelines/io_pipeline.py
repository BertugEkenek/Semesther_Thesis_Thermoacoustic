# pipelines/io_pipeline.py

import os
import numpy as np


def _save_complex_vector(path: str, vec: np.ndarray):
    vec = np.asarray(vec, dtype=np.complex128).ravel()
    np.savetxt(path, vec)


def save_reference_and_noisy_solutions(
    solutions: np.ndarray,
    n_values: np.ndarray,
    noise_sigma: float,
    results_dir: str = "./Results/Solutions/",
):
    """
    Backward compatible single-branch saver.
    """
    os.makedirs(results_dir, exist_ok=True)

    ref_path = os.path.join(results_dir, "Reference_case.txt")
    ref_n_path = os.path.join(results_dir, "Reference_case_n_values.txt")

    noisy_path = os.path.join(results_dir, "Case_with_noise.txt")
    noisy_n_path = os.path.join(results_dir, "Case_with_noise_n_values.txt")

    _save_complex_vector(ref_path, solutions)
    np.savetxt(ref_n_path, np.asarray(n_values).ravel())

    noise_real = np.random.normal(0, noise_sigma, np.asarray(solutions).shape[0])
    noise_imag = np.random.normal(0, noise_sigma, np.asarray(solutions).shape[0])
    noisy_solutions = np.asarray(solutions, dtype=np.complex128) + noise_real + 1j * noise_imag

    _save_complex_vector(noisy_path, noisy_solutions)
    np.savetxt(noisy_n_path, np.asarray(n_values).ravel())

    print("[TXT SAVE] Saved reference and noisy eigenvalues + n_values (single branch)")


def save_reference_and_noisy_solutions_two_branches(
    solutions_branch1: np.ndarray,
    solutions_branch2: np.ndarray,
    n_values: np.ndarray,
    noise_sigma: float,
    results_dir: str = "./Results/Solutions/",
    prefix: str = "Reference_case",
):
    """
    NEW: Save reference and noisy eigenvalues for TWO acoustic branches.

    Files produced:
      - {prefix}_branch1.txt
      - {prefix}_branch2.txt
      - {prefix}_n_values.txt

      - Case_with_noise_branch1.txt
      - Case_with_noise_branch2.txt
      - Case_with_noise_n_values.txt
    """

    os.makedirs(results_dir, exist_ok=True)

    n_values = np.asarray(n_values).ravel()
    b1 = np.asarray(solutions_branch1, dtype=np.complex128).ravel()
    b2 = np.asarray(solutions_branch2, dtype=np.complex128).ravel()

    if b1.shape != b2.shape:
        raise ValueError(f"Branch solution arrays must have same shape. Got {b1.shape} vs {b2.shape}.")
    if n_values.shape[0] != b1.shape[0]:
        raise ValueError(f"n_values length must match solutions length. Got {n_values.shape[0]} vs {b1.shape[0]}.")

    # Reference
    ref_b1_path = os.path.join(results_dir, f"{prefix}_branch1.txt")
    ref_b2_path = os.path.join(results_dir, f"{prefix}_branch2.txt")
    ref_n_path = os.path.join(results_dir, f"{prefix}_n_values.txt")

    _save_complex_vector(ref_b1_path, b1)
    _save_complex_vector(ref_b2_path, b2)
    np.savetxt(ref_n_path, n_values)

    # Noisy
    noisy_b1_path = os.path.join(results_dir, "Case_with_noise_branch1.txt")
    noisy_b2_path = os.path.join(results_dir, "Case_with_noise_branch2.txt")
    noisy_n_path = os.path.join(results_dir, "Case_with_noise_n_values.txt")

    noise_real_1 = np.random.normal(0, noise_sigma, b1.shape[0])
    noise_imag_1 = np.random.normal(0, noise_sigma, b1.shape[0])
    noise_real_2 = np.random.normal(0, noise_sigma, b2.shape[0])
    noise_imag_2 = np.random.normal(0, noise_sigma, b2.shape[0])

    noisy_b1 = b1 + noise_real_1 + 1j * noise_imag_1
    noisy_b2 = b2 + noise_real_2 + 1j * noise_imag_2

    _save_complex_vector(noisy_b1_path, noisy_b1)
    _save_complex_vector(noisy_b2_path, noisy_b2)
    np.savetxt(noisy_n_path, n_values)

    print("[TXT SAVE] Saved reference + noisy eigenvalues for TWO branches + n_values")
