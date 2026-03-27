import keras

def call_func(inputs):
    return keras.ops.arctanh(inputs)

x = keras.random.uniform(shape=(3, 5), minval=-0.5, maxval=0.5)
example_output = call_func(x)