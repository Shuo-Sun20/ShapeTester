import keras

def call_func(inputs, k=0):
    return keras.ops.tril(inputs, k)

example_tensor = keras.random.normal(shape=(3, 3))
example_output = call_func(example_tensor, k=-1)