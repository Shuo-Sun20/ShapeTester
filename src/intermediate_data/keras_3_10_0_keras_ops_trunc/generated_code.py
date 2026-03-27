import keras
from keras import ops

def call_func(inputs):
    return ops.trunc(inputs)

random_tensor = keras.random.uniform(shape=(3, 4), minval=-2.5, maxval=2.5)
example_output = call_func(random_tensor)