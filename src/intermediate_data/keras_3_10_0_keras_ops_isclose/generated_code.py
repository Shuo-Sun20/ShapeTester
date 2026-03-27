import keras

def call_func(inputs, rtol=1e-05, atol=1e-08, equal_nan=False):
    x1, x2 = inputs[0], inputs[1]
    return keras.ops.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)

x1 = keras.random.uniform((3, 4))
x2 = keras.random.uniform((3, 4))
example_output = call_func([x1, x2], rtol=1e-5, atol=1e-8, equal_nan=False)