import keras

def call_func(inputs):
    return keras.ops.log10(inputs)

x = keras.random.uniform(shape=(3, 4), minval=1.0, maxval=100.0)
example_output = call_func(x)