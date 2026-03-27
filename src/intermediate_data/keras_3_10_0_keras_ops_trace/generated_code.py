import keras
import numpy as np

def call_func(inputs, offset=0, axis1=0, axis2=1):
    return keras.ops.trace(x=inputs, offset=offset, axis1=axis1, axis2=axis2)

# Construct a random 4x4 tensor as input
random_tensor = keras.random.normal(shape=(4, 4))
example_output = call_func(inputs=random_tensor, offset=0, axis1=0, axis2=1)