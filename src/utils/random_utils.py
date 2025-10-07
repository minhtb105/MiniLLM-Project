import os
import random
import numpy as np


def set_seed(seed: int):
    """
    Set random seed for Python, NumPy, and PYTHONHASHSEED for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    