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

input_tensor = keras.random.uniform(shape=(2, 16000))
example_output = call_func(
    inputs=input_tensor,
    num_mel_bins=80,
    sampling_rate=8000,
    sequence_stride=128,
    fft_length=2048
)