import keras
import numpy as np

def call_func(inputs):
    return keras.ops.vstack(inputs)

# Generate random tensors for the example
tensor1 = keras.ops.convert_to_tensor(np.random.rand(2, 3))
tensor2 = keras.ops.convert_to_tensor(np.random.rand(1, 3))
example_output = call_func([tensor1, tensor2])