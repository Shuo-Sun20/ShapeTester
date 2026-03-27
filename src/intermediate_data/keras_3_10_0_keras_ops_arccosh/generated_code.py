import keras

def call_func(inputs):
    return keras.ops.arccosh(inputs)

x = keras.random.uniform(shape=(5,), minval=1.0, maxval=10.0)
example_output = call_func(x)