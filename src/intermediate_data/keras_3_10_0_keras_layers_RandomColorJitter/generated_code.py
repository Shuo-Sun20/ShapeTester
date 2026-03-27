import keras
import numpy as np

def call_func(
    value_range=(0, 255),
    brightness_factor=None,
    contrast_factor=None,
    saturation_factor=None,
    hue_factor=None,
    seed=None,
    data_format=None,
    inputs=None
):
    layer = keras.layers.RandomColorJitter(
        value_range=value_range,
        brightness_factor=brightness_factor,
        contrast_factor=contrast_factor,
        saturation_factor=saturation_factor,
        hue_factor=hue_factor,
        seed=seed,
        data_format=data_format
    )
    output = layer(inputs)
    return output

# Construct a valid input
random_input = np.random.rand(2, 224, 224, 3).astype(np.float32) * 255.0
example_output = call_func(inputs=random_input)