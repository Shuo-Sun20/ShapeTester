import keras
import numpy as np

def call_func(inputs):
    x, y = inputs
    return keras.ops.left_shift(x, y)

x_tensor = keras.ops.convert_to_tensor(np.random.randint(1, 10, size=(3, 4), dtype=np.int32))
y_tensor = keras.ops.convert_to_tensor(np.random.randint(0, 4, size=(3, 4), dtype=np.int32))
example_output = call_func([x_tensor, y_tensor])