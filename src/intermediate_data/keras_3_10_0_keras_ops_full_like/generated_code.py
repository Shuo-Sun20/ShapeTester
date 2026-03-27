import keras
import numpy as np

def call_func(inputs, fill_value, dtype=None):
    return keras.ops.full_like(inputs, fill_value, dtype)

random_tensor = keras.random.normal(shape=(3, 4))
example_output = call_func(random_tensor, fill_value=5.0)