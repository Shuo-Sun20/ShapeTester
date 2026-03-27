import keras
import numpy as np

def call_func(
    inputs,
    factor=1.0,
    scale=1.0,
    interpolation="bilinear",
    fill_mode="reflect",
    fill_value=0.0,
    value_range=(0, 255),
    seed=None,
    data_format=None
):
    layer = keras.layers.RandomElasticTransform(
        factor=factor,
        scale=scale,
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=fill_value,
        value_range=value_range,
        seed=seed,
        data_format=data_format
    )
    return layer(inputs)

# Generate a random input tensor
input_tensor = np.random.uniform(0, 255, size=(2, 32, 32, 3)).astype(np.float32)

# Call the function
example_output = call_func(
    inputs=input_tensor,
    factor=0.5,
    scale=(0.8, 1.2),
    interpolation="bilinear",
    fill_mode="reflect",
    fill_value=0.0,
    value_range=(0, 255),
    seed=42,
    data_format=None
)