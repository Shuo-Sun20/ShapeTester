import keras

def call_func(inputs):
    return keras.ops.exp(inputs)

example_input = keras.random.normal((3, 5))
example_output = call_func(example_input)