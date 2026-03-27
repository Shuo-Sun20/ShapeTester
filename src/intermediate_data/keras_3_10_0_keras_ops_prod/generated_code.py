import keras

def call_func(inputs, axis=None, keepdims=False, dtype=None):
    return keras.ops.prod(x=inputs, axis=axis, keepdims=keepdims, dtype=dtype)

example_tensor = keras.random.normal(shape=(3, 4))
example_output = call_func(inputs=example_tensor, axis=1, keepdims=True, dtype="float32")