import keras

def call_func(inputs, ord=None, axis=None, keepdims=False):
    x = inputs[0]
    return keras.ops.norm(x, ord=ord, axis=axis, keepdims=keepdims)

x = keras.random.normal(shape=(3, 4))
example_output = call_func(inputs=[x])