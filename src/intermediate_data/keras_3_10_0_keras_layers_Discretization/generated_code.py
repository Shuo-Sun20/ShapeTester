import keras
import numpy as np

def call_func(bin_boundaries=None, num_bins=None, epsilon=0.01, output_mode="int", sparse=False, dtype=None, name=None, inputs=None):
    layer = keras.layers.Discretization(
        bin_boundaries=bin_boundaries,
        num_bins=num_bins,
        epsilon=epsilon,
        output_mode=output_mode,
        sparse=sparse,
        dtype=dtype,
        name=name
    )
    return layer(inputs)

example_input = np.random.randn(5, 4)
example_output = call_func(bin_boundaries=[-1.0, 0.0, 1.0], inputs=example_input)