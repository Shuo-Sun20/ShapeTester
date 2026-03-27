import keras

def call_func(inputs, offset=0, axis1=0, axis2=1):
    return keras.ops.diagonal(inputs, offset, axis1, axis2)

example_input = keras.random.normal((2, 3, 4))
example_output = call_func(example_input, offset=0, axis1=0, axis2=2)