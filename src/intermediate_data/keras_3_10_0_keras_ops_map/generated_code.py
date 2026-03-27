import keras

def call_func(f, inputs):
    return keras.ops.map(f, inputs)

f = lambda x: x * 2
inputs = keras.random.normal(shape=(5, 3, 3))
example_output = call_func(f, inputs)