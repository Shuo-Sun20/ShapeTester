import keras

def call_func(inputs, axis=None):
    if isinstance(inputs, list):
        x = inputs[0]
    else:
        x = inputs
    return keras.ops.count_nonzero(x, axis)

x = keras.random.uniform((3, 4))
example_output = call_func(x, axis=1)