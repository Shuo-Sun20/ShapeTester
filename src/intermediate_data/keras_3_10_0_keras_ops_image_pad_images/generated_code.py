import keras
import numpy as np

def call_func(
    inputs,
    top_padding=None,
    left_padding=None,
    bottom_padding=None,
    right_padding=None,
    target_height=None,
    target_width=None,
    data_format=None
):
    return keras.ops.image.pad_images(
        images=inputs[0],
        top_padding=top_padding,
        left_padding=left_padding,
        bottom_padding=bottom_padding,
        right_padding=right_padding,
        target_height=target_height,
        target_width=target_width,
        data_format=data_format
    )

random_tensor = np.random.random((2, 15, 25, 3))
example_output = call_func(
    inputs=[random_tensor],
    top_padding=2,
    left_padding=3,
    target_height=20,
    target_width=30
)