import keras

def call_func(inputs):
    return keras.ops.shape(inputs)

x = keras.random.normal((3, 4, 5))
example_output = keras.ops.convert_to_tensor(call_func(x))