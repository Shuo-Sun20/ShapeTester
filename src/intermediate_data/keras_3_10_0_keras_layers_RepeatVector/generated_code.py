import keras

def call_func(n, inputs):
    layer = keras.layers.RepeatVector(n)
    return layer(inputs)

input_tensor = keras.random.normal(shape=(2, 32))
example_output = call_func(3, input_tensor)