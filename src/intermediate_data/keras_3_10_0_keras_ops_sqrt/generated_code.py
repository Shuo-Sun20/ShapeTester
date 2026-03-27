import keras

def call_func(inputs):
    return keras.ops.sqrt(inputs)

example_input = keras.random.normal(shape=(2, 3))
example_output = call_func(example_input)