import keras
import numpy as np

def call_func(inputs):
    x = inputs[0]
    y = inputs[1]
    return keras.ops.bitwise_right_shift(x, y)

x = keras.ops.convert_to_tensor(np.random.randint(0, 256, size=(3, 3), dtype=np.int32))
y = keras.ops.convert_to_tensor(np.random.randint(0, 5, size=(3, 3), dtype=np.int32))
inputs = [x, y]
example_output = call_func(inputs)