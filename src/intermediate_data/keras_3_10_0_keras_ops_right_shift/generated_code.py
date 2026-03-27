import keras
import numpy as np

def call_func(inputs):
    x = inputs[0]
    y = inputs[1]
    return keras.ops.right_shift(x, y)

x = keras.ops.convert_to_tensor(np.random.randint(0, 100, size=(3, 3)), dtype="int32")
y = keras.ops.convert_to_tensor(np.random.randint(0, 5, size=(3, 3)), dtype="int32")
example_output = call_func([x, y])