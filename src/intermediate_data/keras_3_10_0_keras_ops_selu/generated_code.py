import keras
import numpy as np

def call_func(inputs):
    return keras.ops.selu(inputs[0])

random_tensor = np.random.randn(3, 4).astype(np.float32)
example_output = call_func([random_tensor])