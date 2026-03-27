import tensorflow as tf
from dataclasses import dataclass, field

# 1. Define valid_test_case dictionary
valid_test_case = {
    "value_range": (0, 255),
    "num_chains": 3,
    "chain_depth": 3,
    "factor": 0.3,
    "alpha": 1.0,
    "all_ops": True,
    "interpolation": "bilinear",
    "seed": None,
    "data_format": None,
    "inputs": tf.random.uniform(shape=(2, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)
}

# 2. Parameters affecting output shape: data_format only

@dataclass
class InputSpace:
    """
    Class containing all parameters that affect the shape of AugMix output tensor.
    Each parameter's value space is discretized appropriately.
    """
    # data_format parameter affects channel dimension ordering in output shape
    # Discrete parameter with all possible values
    data_format: list = field(default_factory=lambda: [None, "channels_last", "channels_first"])