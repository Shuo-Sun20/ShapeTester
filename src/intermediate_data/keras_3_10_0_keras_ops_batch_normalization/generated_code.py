import keras
import numpy as np

def call_func(inputs, axis, offset=None, scale=None, epsilon=0.001):
    x = inputs[0]
    mean = inputs[1]
    variance = inputs[2]
    return keras.ops.batch_normalization(
        x, mean, variance, axis, offset, scale, epsilon
    )

x = keras.ops.convert_to_tensor(np.random.randn(3, 3))
mean = keras.ops.convert_to_tensor(np.random.randn(3,))
variance = keras.ops.convert_to_tensor(np.random.rand(3,))
example_output = call_func([x, mean, variance], axis=-1)