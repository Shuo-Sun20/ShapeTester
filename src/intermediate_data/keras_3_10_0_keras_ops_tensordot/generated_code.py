import keras

def call_func(inputs, axes=2):
    x1, x2 = inputs
    return keras.ops.tensordot(x1, x2, axes=axes)

x1 = keras.random.normal(shape=(3, 4, 5))
x2 = keras.random.normal(shape=(5, 4, 3))
example_output = call_func(inputs=[x1, x2], axes=1)