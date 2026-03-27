import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List, Union

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [
        tf.constant([1.0, 2.0, 3.0], dtype=tf.float32),  # targets
        tf.constant([0.5, 1.2, -0.3], dtype=tf.float32)  # log_input
    ],
    "compute_full_loss": True,
    "name": "log_poisson_loss_example"
}

# Task 2 & 3: Parameters affecting output shape and their value spaces
# Only 'compute_full_loss' affects values but not shape; shape is determined by inputs tensors.
# However, since we're asked for ALL parameters except 'inputs', we include compute_full_loss and name.
# compute_full_loss: discrete boolean (True, False)
# name: string parameter, discrete set of example values including None

@dataclass
class InputSpace:
    """
    Dataclass containing all parameters of call_func (except 'inputs') 
    that can affect the output tensor's shape (indirectly via value computation).
    """
    compute_full_loss: List[bool] = field(
        default_factory=lambda: [True, False]  # Discrete boolean values
    )
    name: List[Optional[str]] = field(
        default_factory=lambda: [None, "loss_calc_1", "loss_calc_2", "log_poisson_loss", ""]  # Example discrete string values
    )

# Instantiation example (as required)
var = InputSpace()