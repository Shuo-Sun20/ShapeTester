import keras

def call_func(inputs):
    abs_ = inputs[0]
    angle = inputs[1]
    return keras.ops.polar(abs_, angle)

abs_ = keras.random.normal((3, 4))
angle = keras.random.normal((3, 4))
example_output = call_func([abs_, angle])