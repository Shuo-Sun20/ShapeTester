import keras

def call_func(inputs, shape, dtype=None):
    fill_value = inputs[0]
    return keras.ops.full(shape, fill_value, dtype=dtype)

fill_value_tensor = keras.random.uniform(shape=(1,), minval=0, maxval=1)
example_output = call_func([fill_value_tensor], shape=(3, 4, 5), dtype="float32")