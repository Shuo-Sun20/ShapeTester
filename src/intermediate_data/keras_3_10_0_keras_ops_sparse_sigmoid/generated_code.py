import keras
import numpy as np

def call_func(inputs):
    return keras.ops.sparse_sigmoid(inputs)

# Generate random tensor data
random_data = np.random.randn(10).astype(np.float32)
input_tensor = keras.ops.convert_to_tensor(random_data)
example_output = call_func(input_tensor)