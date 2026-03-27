import keras

def call_func(inputs):
    x = inputs[0] if isinstance(inputs, list) else inputs
    return keras.ops.signbit(x)

x = keras.random.uniform(shape=(3, 3), minval=-1, maxval=1)
example_output = call_func([x])