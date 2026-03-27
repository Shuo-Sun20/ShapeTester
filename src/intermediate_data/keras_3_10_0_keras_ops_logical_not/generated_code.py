import keras

def call_func(inputs):
    x = inputs[0]
    output = keras.ops.logical_not(x)
    return output

x = keras.random.normal(shape=(3, 4))
example_output = call_func([x])