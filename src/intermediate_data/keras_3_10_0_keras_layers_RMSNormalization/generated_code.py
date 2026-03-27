import keras
import numpy as np

def call_func(axis, epsilon, inputs):
    layer = keras.layers.RMSNormalization(axis=axis, epsilon=epsilon)
    return layer(inputs)

random_tensor = np.random.rand(1, 10).astype(np.float32)
example_output = call_func(axis=-1, epsilon=1e-06, inputs=random_tensor)