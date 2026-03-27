import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, sequence_length, sequence_stride, fft_length, length=None, window="hann", center=True):
    real_part, imag_part = inputs[0], inputs[1]
    return keras.ops.istft((real_part, imag_part), sequence_length, sequence_stride, fft_length, length, window, center)

# Test with eager tensors (dynamic shape)
real_eager = np.random.rand(2, 10, 9).astype(np.float32)
imag_eager = np.random.rand(2, 10, 9).astype(np.float32)
eager_inputs = [real_eager, imag_eager]

dynamic_output = call_func(eager_inputs, sequence_length=4, sequence_stride=1, fft_length=8, length=1, window="hann", center=False)
print(f"Dynamic output shape: {dynamic_output.shape}")

# Test with Keras.Input placeholders (static shape)
real_input = keras.Input(shape=(10, 9))
imag_input = keras.Input(shape=(10, 9))
static_inputs = [real_input, imag_input]

static_output = call_func(static_inputs, sequence_length=4, sequence_stride=1, fft_length=8, length=1, window="hann", center=False)
print(f"Static output shape: {static_output.shape}")

# Show the defect
print(f"Dynamic shape (eager): [None, {dynamic_output.shape[1]}]")
print(f"Static shape (placeholder): [None, {static_output.shape[1]}]")
print(f"Shapes match: {dynamic_output.shape[1] == static_output.shape[1]}")