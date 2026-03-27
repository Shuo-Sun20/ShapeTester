import keras

def call_func(inputs, axis=-1):
    return keras.ops.sort(inputs, axis=axis)

x = keras.random.uniform(shape=(3, 4, 5))
example_output = call_func(x, axis=-1)