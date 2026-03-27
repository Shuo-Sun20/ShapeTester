import keras
import numpy as np

def call_func(
    height,
    width,
    inputs,
    interpolation="bilinear",
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    fill_mode="constant",
    fill_value=0.0,
    antialias=False,
    data_format=None,
    name=None,
    dtype=None,
    trainable=None
):
    layer = keras.layers.Resizing(
        height=height,
        width=width,
        interpolation=interpolation,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        pad_to_aspect_ratio=pad_to_aspect_ratio,
        fill_mode=fill_mode,
        fill_value=fill_value,
        antialias=antialias,
        data_format=data_format,
        name=name,
        dtype=dtype,
        trainable=trainable
    )
    return layer(inputs)

# Generate random input tensor (batch of 2 images, 64x64, 3 channels, channels_last format)
input_tensor = np.random.rand(2, 64, 64, 3).astype(np.float32)
example_output = call_func(height=32, width=32, inputs=input_tensor, interpolation="bilinear")