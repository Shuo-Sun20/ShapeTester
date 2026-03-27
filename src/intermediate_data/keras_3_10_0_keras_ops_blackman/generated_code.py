import keras

def call_func(inputs):
    return keras.ops.blackman(inputs)

window_length = keras.ops.convert_to_tensor(8)
example_output = call_func(window_length)