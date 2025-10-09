import numpy as np
from typing import Tuple


def _fan_in_out(shape: tuple[int, ...]) -> tuple[int, int]:
    """
    Compute fan_in and fan_out based on the tensor shape.
    
    These values are used to control the variance of the initialization
    distribution for weights to avoid vanishing or exploding gradients.
    
    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the weight tensor. 
        - For Linear layers: (out_features, in_features)
        - For Conv2D layers: (out_channels, in_channels, kernel_h, kernel_w)
    
    Returns
    -------
    (fan_in, fan_out) : tuple[int, int]
        Number of input and output units for a layer.
    """
    if len(shape) == 2:  # Linear layer
        fan_in, fan_out = shape[1], shape[0]
    elif len(shape) == 4:  # Conv2D layer
        receptive_field_size = np.prod(shape[2:])  # kernel_h * kernel_w
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:  # Fallback for uncommon layer shapes
        fan_in = fan_out = int(np.sqrt(np.prod(shape)))
        
    return fan_in, fan_out


# ---------------------------------------------------------------
# Basic initializations
# ---------------------------------------------------------------
def zeros_(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Initialize tensor with all zeros.
    """
    return np.zeros(shape, dtype=np.float32)

def ones_(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Initialize tensor with all ones.
    """
    return np.ones(shape, dtype=np.float32)

def uniform_(shape: Tuple[int, ...], a: float = -0.1, b: float = 0.1) -> np.ndarray:
    """
    Initialize tensor from a uniform distribution U(a, b).
    """
    return np.random.uniform(a, b, size=shape).astype(np.float32)

def normal_(shape: Tuple[int, ...], mean: float = 0.0, std: float = 0.01) -> np.ndarray:
    """
    Initialize tensor from a normal distribution N(mean, std).
    """
    return np.random.normal(mean, std, size=shape).astype(np.float32)


# ---------------------------------------------------------------
# Constant or custom initialization
# ---------------------------------------------------------------

def constant_(shape: Tuple[int, ...], val: float) -> np.ndarray:
    """
    Initialize tensor with a constant value.
    """
    return np.full(shape, val, dtype=np.float32)


def eye_(n: int) -> np.ndarray:
    """
    Identity matrix initialization (useful for RNN orthogonal init).
    """
    return np.eye(n, dtype=np.float32)

# ---------------------------------------------------------------
# Xavier (Glorot) initialization
# ---------------------------------------------------------------

def xavier_uniform_(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Xavier (Glorot) uniform initialization.

    Keeps variance of activations roughly constant across layers.

    Formula:
        limit = sqrt(6 / (fan_in + fan_out))
        W ~ U(-limit, limit)

    Args:
        shape: Shape of tensor to initialize

    Returns:
        np.ndarray with Xavier-initialized values
    """
    fan_in, fan_out = _fan_in_out(shape)
    limit = np.sqrt(6 / (fan_in _ fan_out))
    
    return np.random.uniform(-limit, limit, size=shape).astype(np.float32)

def xavier_normal_(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Xavier (Glorot) normal initialization.

    Formula:
        std = sqrt(2 / (fan_in + fan_out))
        W ~ N(0, std)
    """
    fan_in, fan_out = _fan_in_out(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    
    return np.random.normal(0.0, std, size=shape).astype(np.float32)

# ---------------------------------------------------------------
# Kaiming (He) initialization
# ---------------------------------------------------------------

def kaiming_uniform_(shape: Tuple[int, ...], a: float = 0.0, mode: str = "fan_in") -> np.ndarray:
    """
    Kaiming (He) uniform initialization, recommended for ReLU-based activations.

    Formula:
        bound = sqrt(6 / fan)
        where fan = fan_in (for forward) or fan_out (for backward)

    Args:
        shape: Tensor shape
        a: Negative slope of the activation function (e.g., 0 for ReLU, 0.01 for LeakyReLU)
        mode: 'fan_in' or 'fan_out'

    Returns:
        np.ndarray with Kaiming-initialized values
    """
    fan_in, fan_out = _fan_in_out(shape)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = np.sqrt(2.0 / (1 + a ** 2))
    bound = np.sqrt(3.0) * gain / np.sqrt(fan)
    
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)


def kaiming_normal_(shape: Tuple[int, ...], a: float = 0.0, mode: str = "fan_in") -> np.ndarray:
    """
    Kaiming (He) normal initialization.

    Formula:
        std = gain / sqrt(fan)
        W ~ N(0, std)

    Args:
        shape: Tensor shape
        a: Negative slope for LeakyReLU
        mode: 'fan_in' or 'fan_out'
    """
    fan_in, fan_out = _fan_in_out(shape)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = np.sqrt(2.0 / (1 + a ** 2))
    std = gain / np.sqrt(fan)
    
    return np.random.normal(0.0, std, size=shape).astype(np.float32)
