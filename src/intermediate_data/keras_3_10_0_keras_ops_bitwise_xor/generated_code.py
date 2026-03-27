import keras

def call_func(inputs):
    x, y = inputs[0], inputs[1]
    return keras.ops.bitwise_xor(x, y)

# Generate random integer tensors with shape (2, 3)
x = keras.random.randint(shape=(2, 3), minval=0, maxval=10)
y = keras.random.randint(shape=(2, 3), minval=0, maxval=10)

example_output = call_func([x, y])