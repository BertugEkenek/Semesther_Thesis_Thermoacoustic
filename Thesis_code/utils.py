import logging
import os


def setup_logger(
    name: str,
    logfile: str,
    level: int = logging.INFO,
    fmt: str = "%(message)s",
):
    """
    Create a logger that logs to both console and file.

    Parameters
    ----------
    name : str
        Logger name.
    logfile : str
        Path to log file.
    level : int
        Logging level.
    fmt : str
        Log message format.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Prevent duplicate handlers (important!)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt)

    # --- Console handler ---
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)

    # --- File handler ---
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    fh = logging.FileHandler(logfile, mode="w")
    fh.setFormatter(formatter)
    fh.setLevel(level)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

logger = setup_logger(
        name="mu_fit",
        logfile="./Results/Logs/mu_fit_diagnostics.log"
    )