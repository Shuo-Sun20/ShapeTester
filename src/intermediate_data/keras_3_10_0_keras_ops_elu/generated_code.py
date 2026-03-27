import keras
import numpy as np

def call_func(inputs, alpha=1.0):
    return keras.ops.elu(inputs, alpha)

# Generate a random tensor for demonstration
random_tensor = keras.ops.convert_to_tensor(
    np.random.randn(3, 4).astype(np.float32)
)
example_output = call_func(random_tensor)