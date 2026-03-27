import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import List, Union

def call_func(inputs, adjoint=False, name=None):
    return tf.linalg.inv(input=inputs, adjoint=adjoint, name=name)

# Generate a random 3x3 invertible matrix
np.random.seed(42)
random_matrix = np.random.randn(3, 3).astype(np.float32)
# Ensure it's invertible by making it diagonally dominant
random_matrix = random_matrix + np.eye(3) * 3

valid_test_case = {"inputs": random_matrix, "adjoint": False, "name": None}

@dataclass
class InputSpace:
    # The only parameter that affects output shape (besides inputs) is adjoint
    # adjoint parameter value space (discrete)
    adjoint: List[bool] = None
    
    def __post_init__(self):
        if self.adjoint is None:
            # Discrete parameter space with all possible values
            self.adjoint = [True, False]

# Note: The name parameter does not affect output shape
# The inputs parameter is excluded as per instructions