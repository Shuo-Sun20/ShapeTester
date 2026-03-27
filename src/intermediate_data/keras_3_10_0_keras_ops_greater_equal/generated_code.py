import keras
from keras import ops
import numpy as np

def call_func(inputs):
    return ops.greater_equal(inputs[0], inputs[1])

tensor_a = keras.ops.convert_to_tensor(np.random.rand(3, 4))
tensor_b = keras.ops.convert_to_tensor(np.random.rand(3, 4))
example_output = call_func([tensor_a, tensor_b])