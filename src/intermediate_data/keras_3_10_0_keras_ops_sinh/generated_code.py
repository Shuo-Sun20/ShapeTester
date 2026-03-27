import keras

def call_func(inputs):
    return keras.ops.sinh(inputs)

x = keras.random.normal(shape=(3, 4))
example_output = call_func(x)