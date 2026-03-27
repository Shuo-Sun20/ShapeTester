import keras
import numpy as np

def call_func(inputs, alpha=20.0, sigma=5.0, interpolation="bilinear", fill_mode="reflect", fill_value=0.0, seed=None, data_format=None):
    images = inputs[0]
    return keras.ops.image.elastic_transform(images, alpha, sigma, interpolation, fill_mode, fill_value, seed, data_format)

x = np.random.random((64, 80, 3)).astype("float32")
example_output = call_func([x])