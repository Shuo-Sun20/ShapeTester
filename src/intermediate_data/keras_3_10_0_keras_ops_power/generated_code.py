import keras
import keras.ops as ops

def call_func(inputs):
    x1, x2 = inputs[0], inputs[1]
    return ops.power(x1, x2)

example_output = call_func([keras.random.uniform((3, 4)), keras.random.uniform((3, 4))])