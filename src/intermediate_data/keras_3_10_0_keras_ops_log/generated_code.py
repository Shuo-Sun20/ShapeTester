import keras

def call_func(inputs):
    return keras.ops.log(inputs)

x = keras.random.normal(shape=(2, 3))
example_output = call_func(x)