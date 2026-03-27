import keras
import numpy as np

def call_func(factor, value_range, seed, inputs, training=False):
    layer = keras.layers.RandomBrightness(factor=factor, value_range=value_range, seed=seed)
    output = layer(inputs, training=training)
    return output

np.random.seed(42)
random_tensor = np.random.uniform(low=0.0, high=255.0, size=(2, 2, 3))
example_output = call_func(factor=0.2, value_range=(0, 255), seed=42, inputs=random_tensor, training=True)