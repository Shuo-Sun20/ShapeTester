import keras
import numpy as np

def call_func(rate, inputs, seed=None, name=None, dtype=None, training=False):
    layer = keras.layers.SpatialDropout1D(rate=rate, seed=seed, name=name, dtype=dtype)
    return layer(inputs, training=training)

input_tensor = keras.random.normal(shape=(2, 10, 5))
example_output = call_func(rate=0.5, inputs=input_tensor, seed=42, training=True)