import keras
import numpy as np

def call_func(inputs, sequence_length, sequence_stride, fft_length, length=None, window="hann", center=True):
    real_part, imag_part = inputs[0], inputs[1]
    return keras.ops.istft((real_part, imag_part), sequence_length, sequence_stride, fft_length, length, window, center)

batch_size = 2
time_frames = 10
freq_bins = 9
random_real = keras.ops.convert_to_tensor(np.random.randn(batch_size, time_frames, freq_bins).astype(np.float32))
random_imag = keras.ops.convert_to_tensor(np.random.randn(batch_size, time_frames, freq_bins).astype(np.float32))
example_output = call_func([random_real, random_imag], 4, 2, 8, None, "hann", True)