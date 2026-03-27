import keras
import numpy as np

def call_func(inputs, height_factor, width_factor=None, fill_mode="reflect", interpolation="bilinear", seed=None, fill_value=0.0, data_format=None):
    layer = keras.layers.RandomZoom(
        height_factor=height_factor,
        width_factor=width_factor,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=data_format
    )
    return layer(inputs)

random_tensor = np.random.rand(5, 224, 224, 3)
example_output = call_func(random_tensor, height_factor=(0.2, 0.3), width_factor=(-0.2, 0.2), fill_mode="reflect", interpolation="bilinear", seed=42)