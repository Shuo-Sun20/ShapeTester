import keras
import numpy as np

def call_func(stddev, inputs, training, seed=None):
    layer = keras.layers.GaussianNoise(stddev=stddev, seed=seed)
    return layer(inputs, training=training)

np.random.seed(42)
example_input = np.random.randn(4, 32, 32, 3).astype('float32')
example_output = call_func(stddev=0.1, inputs=example_input, training=True)