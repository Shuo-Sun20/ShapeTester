import keras
import numpy as np

def call_func(inputs, pattern, axes_lengths=None):
    if axes_lengths is None:
        axes_lengths = {}
    return keras.ops.rearrange(inputs, pattern, **axes_lengths)

images = np.random.rand(32, 30, 40, 3)
example_output = call_func(images, 'b h w c -> b c h w')