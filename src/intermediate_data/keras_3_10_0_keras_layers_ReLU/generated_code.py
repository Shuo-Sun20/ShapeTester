import keras
import numpy as np

def call_func(max_value=None, negative_slope=0.0, threshold=0.0, name=None, dtype=None, inputs=None):
    layer = keras.layers.ReLU(max_value=max_value, negative_slope=negative_slope, threshold=threshold, name=name, dtype=dtype)
    return layer(inputs)

random_input = np.random.randn(5, 3)
example_output = call_func(max_value=10, negative_slope=0.5, threshold=0, inputs=random_input)