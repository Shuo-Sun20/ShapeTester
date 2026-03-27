import keras

def call_func(inputs, fft_length=None):
    result = keras.ops.rfft(inputs, fft_length)
    return keras.ops.stack(result, axis=0)

x = keras.random.normal(shape=(10,))
example_output = call_func(x)