import keras
import numpy as np

def call_func(inputs, mode="reduced"):
    q, r = keras.ops.qr(x=inputs, mode=mode)
    return [q, r]

example_input = keras.ops.convert_to_tensor(np.random.randn(3, 2).astype(np.float32))
example_output = call_func(example_input)