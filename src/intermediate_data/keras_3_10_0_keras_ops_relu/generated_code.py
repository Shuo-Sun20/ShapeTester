import keras
import numpy as np

def call_func(inputs):
    return keras.ops.relu(inputs[0])

random_tensor = keras.random.normal(shape=(4, 3))
example_output = call_func([random_tensor])