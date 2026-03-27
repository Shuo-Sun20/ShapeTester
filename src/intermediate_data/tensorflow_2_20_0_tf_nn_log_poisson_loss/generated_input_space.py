import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [
        tf.constant([1.0, 2.0, 3.0], dtype=tf.float32),
        tf.constant([0.5, 1.2, -0.3], dtype=tf.float32)
    ],
    "compute_full_loss": True,
    "name": "log_poisson_loss_example"
}

# Tasks 2, 3 & 4: Define InputSpace with parameters affecting output shape
@dataclass
class InputSpace:
    # Parameters that affect output shape (besides inputs):
    # 1. compute_full_loss: boolean parameter that affects the computation
    #    Discrete parameter with all possible values
    compute_full_loss: List[bool] = field(default_factory=lambda: [True, False])
    
    # 2. name: string parameter (can affect operation naming but not shape)
    #    Discrete parameter with typical values including None and example names
    name: List[Optional[str]] = field(default_factory=lambda: [
        None,
        "log_poisson_loss_example",
        "log_poisson_loss_op",
        "custom_loss_name",
        "test_loss_operation",
        ""
    ])