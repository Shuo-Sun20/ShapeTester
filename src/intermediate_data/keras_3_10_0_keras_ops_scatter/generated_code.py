import keras
import numpy as np

def call_func(inputs, shape):
    indices, values = inputs
    return keras.ops.scatter(indices, values, shape)

# Generate random indices and values
np.random.seed(42)
indices = np.random.randint(0, 3, size=(5, 2)).tolist()  # 5 pairs of 2D indices
values = np.random.random(size=5).astype(np.float32)     # 5 random float values
shape = (4, 4)

example_output = call_func([indices, values], shape)