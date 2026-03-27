import keras
import numpy as np

def call_func(l1, l2, inputs):
    layer = keras.layers.ActivityRegularization(l1=l1, l2=l2)
    return layer(inputs)

random_input = np.random.randn(2, 5, 8).astype('float32')
example_output = call_func(l1=0.01, l2=0.02, inputs=random_input)