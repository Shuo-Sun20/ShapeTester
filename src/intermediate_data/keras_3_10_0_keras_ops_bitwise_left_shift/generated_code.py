import keras
import numpy as np

def call_func(inputs):
    x, y = inputs
    return keras.ops.bitwise_left_shift(x, y)

x_tensor = keras.ops.convert_to_tensor(np.random.randint(0, 100, size=(3, 3)))
y_tensor = keras.ops.convert_to_tensor(np.random.randint(0, 8, size=(3, 3)))
example_output = call_func([x_tensor, y_tensor])