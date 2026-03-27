import keras

def call_func(inputs):
    return keras.ops.sign(inputs)

example_input = keras.random.normal((3, 4))
example_output = call_func(example_input)