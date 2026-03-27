import keras

def call_func(inputs, k=0):
    return keras.ops.diagflat(x=inputs, k=k)

x = keras.random.normal(shape=(4,))
example_output = call_func(inputs=x, k=0)