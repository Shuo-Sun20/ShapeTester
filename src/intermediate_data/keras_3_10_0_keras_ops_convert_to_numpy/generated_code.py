import keras
import keras.ops as ops

def call_func(inputs):
    return ops.convert_to_numpy(inputs[0])

tensor = keras.random.normal(shape=(3, 4))
example_output = call_func([tensor])