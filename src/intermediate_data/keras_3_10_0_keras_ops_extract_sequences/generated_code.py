import keras

def call_func(inputs, sequence_length, sequence_stride):
    return keras.ops.extract_sequences(inputs, sequence_length, sequence_stride)

x = keras.random.uniform(shape=(100,))
example_output = call_func(x, 5, 2)