import keras
import numpy as np

def call_func(
    height_factor,
    width_factor,
    inputs,
    fill_mode="reflect",
    interpolation="bilinear",
    seed=None,
    fill_value=0.0,
    data_format=None
):
    layer = keras.layers.RandomTranslation(
        height_factor=height_factor,
        width_factor=width_factor,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=data_format
    )
    return layer(inputs)

example_input = np.random.rand(4, 32, 32, 3).astype(np.float32)
example_output = call_func(
    height_factor=0.2,
    width_factor=(-0.1, 0.1),
    inputs=example_input,
    fill_mode="constant",
    interpolation="bilinear",
    seed=42,
    fill_value=0.5,
    data_format="channels_last"
)