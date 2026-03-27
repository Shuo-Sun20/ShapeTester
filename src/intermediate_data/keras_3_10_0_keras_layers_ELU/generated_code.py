import keras
import numpy as np

def call_func(inputs, alpha=1.0, name=None, dtype=None):
    elu_instance = keras.layers.ELU(alpha=alpha, name=name, dtype=dtype)
    return elu_instance(inputs)

# Generate random input tensor
random_tensor = np.random.randn(2, 5, 10).astype(np.float32)
example_output = call_func(random_tensor, alpha=1.0)