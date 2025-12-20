import numpy as np
import scipy.io as sio
from utils import logger



def load_data(
    data_path: str,
    show_position: bool = False,
    show_max_min: bool = False
):
    # Load
    mat = sio.loadmat(data_path, squeeze_me=False, struct_as_record=False)
    data = {k: v for k, v in mat.items() if not k.startswith("__")}

    for k in ("n_vec","R_vec", "EV"):
        if k not in data:
            logger.warning(f"Missing key in data: {k}")

    # Convert to numpy arrays
    n = np.asanyarray(data["n_vec"]).squeeze()
    R = np.asanyarray(data["R_vec"]).squeeze()

    # EV shape checks
    EV = np.asarray(data["EV"], dtype=object)

    EV0 = EV[:, 0, 0]
    EV_trajectories = EV[:, 0:, 1:]
    
    # Analyze the trajectories sizes
    smaller_entries, max_size, min_size = analyze_trajectories_sizes(EV_trajectories, show_position=show_position, show_max_min=show_max_min)
    
    return n, R, EV0, EV_trajectories, smaller_entries, max_size, min_size


def analyze_trajectories_sizes(trajectories, show_position=False, show_max_min=False):
    """
    Analyze the sizes of trajectories and identify entries smaller than the maximum size.
    
    Args:
        trajectories: 3D numpy array of object arrays to analyze
        
    Returns:
        tuple: (smaller_entries, max_size, min_size)
            - smaller_entries: list of tuples (i,j,k,size) for entries smaller than max_size
            - max_size: maximum size found across all entries
            - min_size: minimum size found across all entries
    """
    max_size = 0
    min_size = float('inf')
    smaller_entries = []  # Will store tuples of (i,j,k,size) for entries smaller than max_size
    
    # First pass to find max_size
    for i in range(trajectories.shape[0]):
        for j in range(trajectories.shape[1]):
            for k in range(trajectories.shape[2]):
                entry = trajectories[i,j,k]
                if entry.size == 0:  # Skip empty arrays
                    continue
                squeezed = entry.squeeze()
                # Handle both 1D and multi-dimensional arrays
                current_size = squeezed.shape[0] if len(squeezed.shape) > 0 else 0
                if current_size > 0:  # Only consider non-empty arrays
                    max_size = max(max_size, current_size)
                    min_size = min(min_size, current_size)
    
    # Second pass to collect smaller entries
    for i in range(trajectories.shape[0]):
        for j in range(trajectories.shape[1]):
            for k in range(trajectories.shape[2]):
                entry = trajectories[i,j,k]
                if entry.size == 0:  # Skip empty arrays
                    continue
                squeezed = entry.squeeze()
                current_size = squeezed.shape[0] if len(squeezed.shape) > 0 else 0
                if 0 < current_size < max_size:
                    smaller_entries.append((i, j, k, current_size))
    if show_max_min:
        logger.info("\nChecking sizes ...")
        logger.info(f"Maximum size: {max_size}")
        logger.info(f"Minimum size: {min_size}")
        logger.info(f"Found {len(smaller_entries)} entries smaller than maximum size")
    if show_position and smaller_entries:
        logger.info("\nSmaller entries (i,j,k,size):")
        for entry in smaller_entries:
            logger.info(f"    Position ({entry[0]},{entry[1]},{entry[2]}): {entry[3]}")

    return smaller_entries, max_size, min_size

def reshape_EV_trajectories(EV_trajectories, min_size):
    """
    Reshape all entries in EV_trajectories to match the minimum size.
    This function truncates larger arrays to match the minimum size found across all entries.
    
    Args:
        EV_trajectories: The input array to reshape
        min_size: The minimum size to reshape all entries to
    
    Returns:
        numpy.ndarray: The reshaped array
    """
    shape = EV_trajectories.shape
    reshaped_data = np.empty(shape, dtype=object)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                current = EV_trajectories[i,j,k]
                if current.size == 0:  # Skip empty arrays
                    reshaped_data[i,j,k] = np.array([])
                    continue
                    
                squeezed = current.squeeze()
                if len(squeezed.shape) == 0:  # Handle scalar values
                    reshaped_data[i,j,k] = squeezed
                    continue
                    
                current_size = squeezed.shape[0]
                if current_size > min_size:
                    reshaped_data[i,j,k] = squeezed[:min_size]
                else:
                    reshaped_data[i,j,k] = squeezed
                    
    return reshaped_data

def load_experimental_data(
    experimental_data_path: str,
    logger: object = logger
):

    mat = sio.loadmat(experimental_data_path, squeeze_me=False, struct_as_record=False)
    data = {k: v for k, v in mat.items() if not k.startswith("__")}

    for k in ("n_vec","R_vec", "EV"):
        if k not in data:
            logger.warning(f"Missing key in data: {k}")

    EV = np.asarray(data["EV"], dtype=object)
    EV_experimental = np.empty((EV.shape[2]-1, EV[0,0,1].shape[1]), dtype=complex)
    EV_experimental0 = EV[0,0,0]
    for i in range(1,EV_experimental.shape[0]+1):
        for j in range(EV_experimental.shape[1]):
            EV_experimental[i-1, j] = EV[0,0,i].squeeze()[j]
    return EV_experimental, EV_experimental0

def load_txt_solutions(txt_path: str):
    """
    Correct, robust TXT loader.
    Ensures:
      - complex parsing correct
      - sigma = real part
      - omega = imag part
      - consistent shapes with MAT loader
    """

    # Load roots as complex numbers
    raw = np.loadtxt(txt_path, dtype=str)
    EV_list = np.array([complex(x.replace('i','j')) for x in raw.ravel()])
    N = EV_list.shape[0]

    if N < 2:
        raise ValueError("TXT file must contain at least 2 eigenvalues.")

    # Matching n values
    n_file = f"./Results/Solutions/Reference_case_n_values.txt"
    n = np.loadtxt(n_file).ravel()
    if len(n) != N:
        raise ValueError("Mismatch: n-values and EV_list sizes differ.")

    # One R only
    R = np.array([-1.0])

    # EV0 = first eigenvalue
    EV0 = np.empty((1,1,1,1), dtype=object)
    EV0[0,0,0,0] = EV_list[0]

    # Trajectory = remaining eigenvalues
    m = N-1
    EV_trajectories = np.empty((1,1,m,1), dtype=object)
    for k in range(m):
        EV_trajectories[0,0,k,0] = EV_list[k+1]

    # min_size = m for consistency
    return n, R, EV0, EV_trajectories, [], m, m
