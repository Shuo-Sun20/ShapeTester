import keras

def call_func(inputs, mode="valid"):
    x1 = inputs[0]
    x2 = inputs[1]
    return keras.ops.correlate(x1, x2, mode=mode)

# Generate random 1D tensors
M, N = 5, 3
x1 = keras.random.uniform((M,))
x2 = keras.random.uniform((N,))
example_output = call_func([x1, x2], mode="valid")