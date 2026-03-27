import keras
import numpy as np

def call_func(inputs):
    x1, x2 = inputs[0], inputs[1]
    return keras.ops.logical_or(x1, x2)

tensor1 = keras.random.normal(shape=(3, 4))
tensor2 = keras.random.normal(shape=(3, 4))
example_output = call_func([tensor1, tensor2])