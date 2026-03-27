import keras

def call_func(inputs):
    x1, x2 = inputs[0], inputs[1]
    return keras.ops.outer(x1, x2)

# Create random 1D tensors for example inputs
x1_example = keras.random.uniform(shape=(3,))
x2_example = keras.random.uniform(shape=(4,))

example_output = call_func([x1_example, x2_example])