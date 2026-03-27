import keras

def call_func(inputs):
    return keras.ops.arccos(inputs)

random_tensor = keras.random.uniform(shape=(3, 4), minval=-1, maxval=1)
example_output = call_func(random_tensor)