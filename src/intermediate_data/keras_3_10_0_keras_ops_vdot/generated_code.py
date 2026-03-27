import keras
import numpy as np

def call_func(inputs):
    x1 = inputs[0]
    x2 = inputs[1]
    return keras.ops.vdot(x1, x2)

x1 = keras.ops.convert_to_tensor(np.random.randn(3, 4))
x2 = keras.ops.convert_to_tensor(np.random.randn(3, 4))
example_output = call_func([x1, x2])