import keras

def call_func(inputs, dtype):
    return keras.ops.cast(x=inputs, dtype=dtype)

x = keras.random.normal(shape=(3, 4))
example_output = call_func(inputs=x, dtype="float16")