import keras
import numpy as np

def call_func(inputs, shape):
    return list(keras.ops.unravel_index(indices=inputs, shape=shape))

np.random.seed(42)
indices_tensor = keras.ops.convert_to_tensor(
    np.random.randint(0, 12, size=(4,)), dtype="int32"
)
shape_tuple = (3, 4)

example_output = call_func(indices_tensor, shape_tuple)