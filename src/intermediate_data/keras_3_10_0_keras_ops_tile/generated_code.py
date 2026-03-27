import keras

def call_func(inputs, repeats):
    return keras.ops.tile(inputs, repeats)

x = keras.random.normal(shape=(2, 3))
repeats = (2, 3)
example_output = call_func(x, repeats)