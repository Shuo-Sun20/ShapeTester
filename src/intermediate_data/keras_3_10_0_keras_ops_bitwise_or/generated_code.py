import keras
import numpy as np

def call_func(inputs):
    x, y = inputs[0], inputs[1]
    return keras.ops.bitwise_or(x, y)

x = keras.ops.convert_to_tensor(np.random.randint(0, 100, (3, 4)))
y = keras.ops.convert_to_tensor(np.random.randint(0, 100, (3, 4)))
example_output = call_func([x, y])