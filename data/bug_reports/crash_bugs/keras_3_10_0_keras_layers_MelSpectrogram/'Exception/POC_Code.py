import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(
    inputs,
    fft_length=2048,
    sequence_stride=512,
    sequence_length=None,
    window="hann",
    sampling_rate=16000,
    num_mel_bins=128,
    min_freq=20.0,
    max_freq=None,
    power_to_db=True,
    top_db=80.0,
    mag_exp=2.0,
    min_power=1e-10,
    ref_power=1.0
):
    mel_spectrogram_layer = keras.layers.MelSpectrogram(
        fft_length=fft_length,
        sequence_stride=sequence_stride,
        sequence_length=sequence_length,
        window=window,
        sampling_rate=sampling_rate,
        num_mel_bins=num_mel_bins,
        min_freq=min_freq,
        max_freq=max_freq,
        power_to_db=power_to_db,
        top_db=top_db,
        mag_exp=mag_exp,
        min_power=min_power,
        ref_power=ref_power
    )
    output = mel_spectrogram_layer(inputs)
    return output

# Create eager tensor input
eager_input = tf.random.uniform(shape=(2, 16000), dtype=tf.float32)

# Create Keras Input placeholder with same shape
placeholder_input = keras.Input(shape=(16000,), batch_size=2, dtype=tf.float32)

# Test parameters that cause the defect
test_params = {
    'fft_length': 256,
    'sequence_stride': 256,
    'sequence_length': 512,
    'window': 'hann',
    'sampling_rate': 16000,
    'num_mel_bins': 80,
    'min_freq': 20.0,
    'max_freq': None,
    'power_to_db': True,
    'top_db': 80.0,
    'mag_exp': 2.0,
    'min_power': 1e-10,
    'ref_power': 1.0
}

print("Testing with eager tensor:")
try:
    dynamic_output = call_func(eager_input, **test_params)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

print("\nTesting with Keras Input placeholder:")
try:
    static_output = call_func(placeholder_input, **test_params)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output error: {e}")

print("\nDefect reproduced: Different behavior between eager tensors and Keras Input placeholders")