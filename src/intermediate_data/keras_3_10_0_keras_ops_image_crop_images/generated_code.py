import keras
import numpy as np

def call_func(
    inputs,
    top_cropping=None,
    left_cropping=None,
    bottom_cropping=None,
    right_cropping=None,
    target_height=None,
    target_width=None,
    data_format=None
):
    return keras.ops.image.crop_images(
        images=inputs,
        top_cropping=top_cropping,
        left_cropping=left_cropping,
        bottom_cropping=bottom_cropping,
        right_cropping=right_cropping,
        target_height=target_height,
        target_width=target_width,
        data_format=data_format
    )

random_images = np.random.randn(4, 32, 32, 3).astype("float32")
example_output = call_func(
    inputs=random_images,
    top_cropping=2,
    left_cropping=3,
    bottom_cropping=4,
    right_cropping=5,
    data_format="channels_last"
)