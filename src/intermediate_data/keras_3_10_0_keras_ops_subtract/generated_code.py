import keras
import numpy as np

def call_func(inputs):
    x1, x2 = inputs[0], inputs[1]
    return keras.ops.subtract(x1, x2)

tensor1 = keras.ops.convert_to_tensor(np.random.rand(3, 4))
tensor2 = keras.ops.convert_to_tensor(np.random.rand(3, 4))
example_output = call_func([tensor1, tensor2])