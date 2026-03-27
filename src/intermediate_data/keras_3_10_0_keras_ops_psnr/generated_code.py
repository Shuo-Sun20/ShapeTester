import keras

def call_func(inputs, max_val):
    x1, x2 = inputs[0], inputs[1]
    return keras.ops.psnr(x1, x2, max_val)

x1 = keras.random.normal((2, 4, 4, 3))
x2 = keras.random.normal((2, 4, 4, 3))
example_output = call_func([x1, x2], 1.0)