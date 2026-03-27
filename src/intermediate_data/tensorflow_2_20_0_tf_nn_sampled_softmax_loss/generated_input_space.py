import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# Task 1: Define valid_test_case dictionary
valid_test_case = {
    "inputs": [
        tf.random.normal(shape=[10000, 128]),  # weights [num_classes, dim]
        tf.random.normal(shape=[10000]),       # biases [num_classes]
        tf.random.uniform(shape=[32, 2], minval=0, maxval=10000, dtype=tf.int64),  # labels [batch_size, num_true]
        tf.random.normal(shape=[32, 128])      # network_inputs [batch_size, dim]
    ],
    "num_sampled": 100,
    "num_classes": 10000,
    "num_true": 2,
    "sampled_values": None,
    "remove_accidental_hits": True,
    "seed": None,
    "name": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters affecting output shape (batch_size dimension)
    num_sampled: List[int] = field(default_factory=lambda: [1, 10, 50, 100, 500])
    num_classes: List[int] = field(default_factory=lambda: [1000, 5000, 10000, 20000, 50000])
    num_true: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    remove_accidental_hits: List[bool] = field(default_factory=lambda: [True, False])

# Note: The output shape is determined by batch_size in inputs, 
# but since "inputs" is excluded per instructions, these are the remaining 
# parameters that affect the computation and must be properly configured.