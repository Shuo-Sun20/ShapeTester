import keras
import numpy as np

def call_func(inputs):
    x1, x2 = inputs
    return keras.ops.logaddexp(x1, x2)

tensor1 = keras.ops.convert_to_tensor(np.random.randn(3, 4))
tensor2 = keras.ops.convert_to_tensor(np.random.randn(3, 4))
example_output = call_func([tensor1, tensor2])