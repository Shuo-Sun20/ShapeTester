import keras

def call_func(inputs):
    return keras.ops.erf(inputs)

example_output = call_func(keras.random.normal((3, 4)))