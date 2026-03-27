import numpy as np
import keras

def call_func(inputs):
    x1 = inputs[0]
    x2 = inputs[1]
    return keras.ops.arctan2(x1, x2)

x1_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4))
x2_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4))
example_output = call_func([x1_tensor, x2_tensor])