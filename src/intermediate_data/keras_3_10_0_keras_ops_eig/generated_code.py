import keras

def call_func(inputs):
    return keras.ops.eig(inputs[0])

example_input = keras.random.normal(shape=(4, 4))
example_output = list(call_func([example_input]))