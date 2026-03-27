import keras
import keras.ops as ops

def call_func(inputs, n=1, axis=-1):
    a = inputs[0] if isinstance(inputs, list) else inputs
    return ops.diff(a, n=n, axis=axis)

random_tensor = keras.random.uniform(shape=(5, 3))
example_output = call_func(random_tensor, n=1, axis=-1)