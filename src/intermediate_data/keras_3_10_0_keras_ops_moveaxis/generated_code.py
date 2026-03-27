import keras
import numpy as np

def call_func(inputs, source, destination):
    return keras.ops.moveaxis(x=inputs, source=source, destination=destination)

x = keras.random.normal(shape=(3, 4, 5))
example_output = call_func(inputs=x, source=0, destination=-1)