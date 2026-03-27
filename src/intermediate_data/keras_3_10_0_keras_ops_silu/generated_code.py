import keras

def call_func(inputs):
    return keras.ops.silu(inputs)

x = keras.random.uniform(shape=(5,), minval=-10.0, maxval=10.0)
example_output = call_func(x)