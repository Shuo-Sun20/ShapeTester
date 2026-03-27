import keras

def call_func(N, M=None, k=0, dtype=None, inputs=None):
    return keras.ops.tri(N, M, k, dtype)

example_output = call_func(N=5, M=5, k=0, dtype="float32", inputs=None)