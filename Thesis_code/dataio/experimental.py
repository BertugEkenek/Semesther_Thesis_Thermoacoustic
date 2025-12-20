import numpy as np
from scipy.io import loadmat
import logging

def overlay_experimental_eigenvalues(ax, mat_file: str, omega_ref: float, window: float = 500):
    """
    Overlay experimental eigenvalues from a .mat file onto the plot.

    Parameters:
    - ax: Matplotlib Axes object
    - mat_file: path to the .mat file containing 'EV'
    - omega_ref: reference frequency (rad/s) to center filtering
    - window: frequency bandwidth for filtering
    """
    try:
        mat_data = loadmat(mat_file)
        EV_raw = mat_data.get("EV")
        if EV_raw is None:
            logging.warning("EV data not found in .mat file.")
            return

        s_vals_clean = np.concatenate([
            block.flatten() for block in EV_raw.ravel()
            if isinstance(block, np.ndarray)
        ])

        sigma_vals = np.real(s_vals_clean)
        omega_vals = np.imag(s_vals_clean)

        mask = (
            (omega_vals > omega_ref - window*2) &
            (omega_vals < omega_ref + window*2) &
            (np.abs(sigma_vals) < 1000)            &
            (omega_vals < 4000) # Hard cutoff for better visualization

        )

        ax.scatter(omega_vals[mask], sigma_vals[mask], color='red', marker='x', label='taX Data')

    except Exception as e:
        logging.error(f"Failed to overlay experimental eigenvalues: {e}")

def overlay_reference_eigenvalues(ax, txt_file: str, omega_ref: float, window: float = 500):
    """
    Overlay reference eigenvalues from a .txt file containing complex numbers onto the plot.

    Parameters:
    - ax: Matplotlib Axes object
    - txt_file: path to the .txt file containing complex eigenvalues
    - omega_ref: reference frequency (rad/s) to center filtering
    - window: frequency bandwidth for filtering
    """
    try:
        # Load data as strings and convert to complex
        raw_data = np.loadtxt(txt_file, dtype=str)
        complex_vals = np.array([complex(val.replace('i', 'j')) for val in raw_data.flatten()])

        sigma_vals = np.real(complex_vals)
        omega_vals = np.imag(complex_vals)

        # Apply filtering
        mask = (
            (omega_vals > omega_ref - window*2) &
            (omega_vals < omega_ref + window*2) &
            (np.abs(sigma_vals) < 1000) &   # optional cutoff
            (omega_vals < 3000)             # optional cutoff
        )

        ax.scatter(omega_vals[mask], sigma_vals[mask], color='green', marker='+', label='Reference Data')

    except Exception as e:
        logging.error(f"Failed to overlay reference eigenvalues: {e}")
