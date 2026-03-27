import keras
import numpy as np

def call_func(inputs, k, sorted=True):
    values, indices = keras.ops.top_k(inputs, k, sorted)
    return [values, indices]

random_tensor = keras.random.normal(shape=(10,))
example_output = call_func(random_tensor, k=3)