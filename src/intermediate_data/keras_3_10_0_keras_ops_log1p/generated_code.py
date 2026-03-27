import keras

def call_func(inputs):
    x = inputs[0]
    return keras.ops.log1p(x)

x = keras.random.uniform(shape=(3, 4))
example_output = call_func([x])