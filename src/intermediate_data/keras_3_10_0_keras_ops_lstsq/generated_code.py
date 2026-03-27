import keras

def call_func(inputs, rcond=None):
    a, b = inputs
    return keras.ops.lstsq(a, b, rcond=rcond)

# Generate random input tensors
M, N, K = 6, 4, 2
a = keras.random.normal(shape=(M, N))
b = keras.random.normal(shape=(M, K))

# Call function and store output
example_output = call_func([a, b])