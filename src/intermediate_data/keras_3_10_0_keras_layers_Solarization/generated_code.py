import keras
import numpy as np

def call_func(
    inputs,
    addition_factor=0.0,
    threshold_factor=0.0,
    value_range=(0, 255),
    seed=None,
):
    solarization_layer = keras.layers.Solarization(
        addition_factor=addition_factor,
        threshold_factor=threshold_factor,
        value_range=value_range,
        seed=seed,
    )
    output = solarization_layer(inputs)
    return output

batch_size = 4
height = 32
width = 32
channels = 3
random_images = np.random.randint(
    low=0,
    high=256,
    size=(batch_size, height, width, channels),
    dtype=np.uint8,
)
example_output = call_func(
    inputs=random_images,
    addition_factor=0.25,
    threshold_factor=0.5,
    value_range=(0, 255),
    seed=42,
)