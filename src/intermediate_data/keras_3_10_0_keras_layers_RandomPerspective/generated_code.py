import keras
import numpy as np

def call_func(
    inputs,
    factor=1.0,
    scale=1.0,
    interpolation="bilinear",
    fill_value=0.0,
    seed=None,
    data_format=None
):
    layer = keras.layers.RandomPerspective(
        factor=factor,
        scale=scale,
        interpolation=interpolation,
        fill_value=fill_value,
        seed=seed,
        data_format=data_format
    )
    return layer(inputs)

random_tensor = np.random.rand(2, 32, 32, 3).astype(np.float32)
example_output = call_func(random_tensor, factor=0.5, scale=0.3)