import tensorflow as tf
from dataclasses import dataclass

valid_test_case = {
    'inputs': tf.random.normal(shape=(2, 3)),
    'axis': 1,
    'keepdims': True,
    'name': None
}

@dataclass
class InputSpace:
    axis: list = None
    keepdims: list = None
    
    def __post_init__(self):
        if self.axis is None:
            # Example discrete values for axis parameter (assuming input tensor rank <= 4)
            self.axis = [
                None,          # Reduce all dimensions
                0,             # Reduce along first dimension
                1,             # Reduce along second dimension  
                [0, 1],        # Reduce along first and second dimensions
                [-1],          # Reduce along last dimension
                [-2, -1]       # Reduce along last two dimensions
            ]
        if self.keepdims is None:
            self.keepdims = [True, False]