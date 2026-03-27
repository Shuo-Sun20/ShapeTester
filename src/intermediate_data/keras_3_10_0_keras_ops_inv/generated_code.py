import keras

def call_func(inputs):
    return keras.ops.inv(inputs)

x = keras.random.normal(shape=(3, 3))
example_output = call_func(x)