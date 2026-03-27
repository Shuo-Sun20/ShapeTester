import keras
from keras import ops

def call_func(inputs, indices, axis=None):
    return ops.take_along_axis(x=inputs, indices=indices, axis=axis)

x = keras.random.normal(shape=(3, 4, 5))
indices = keras.random.randint(shape=(3, 2, 5), minval=0, maxval=4)
example_output = call_func(inputs=x, indices=indices, axis=1)