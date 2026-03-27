import keras
import numpy as np

def call_func(factor, value_range, seed, inputs, training=None):
    layer_instance = keras.layers.RandomContrast(factor=factor, value_range=value_range, seed=seed)
    output_tensor = layer_instance(inputs, training=training)
    return output_tensor

input_tensor = np.random.uniform(low=0.0, high=255.0, size=(4, 32, 32, 3)).astype('float32')
example_output = call_func(factor=0.5, value_range=(0, 255), seed=42, inputs=input_tensor, training=True)