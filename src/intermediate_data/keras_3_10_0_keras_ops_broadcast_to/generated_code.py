import keras

def call_func(inputs, shape):
    x = inputs[0]
    return keras.ops.broadcast_to(x, shape)

x = keras.random.normal((3,))
example_output = call_func([x], (2, 3))