import keras
import numpy as np

def call_func(inputs, axis=0):
    return keras.ops.concatenate(xs=inputs, axis=axis)

tensor1 = keras.ops.convert_to_tensor(np.random.randn(3, 4, 5))
tensor2 = keras.ops.convert_to_tensor(np.random.randn(3, 4, 5))
tensor3 = keras.ops.convert_to_tensor(np.random.randn(3, 4, 5))

example_output = call_func(inputs=[tensor1, tensor2, tensor3], axis=0)