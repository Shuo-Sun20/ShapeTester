import keras

def call_func(inputs):
    x1, x2 = inputs[0], inputs[1]
    return keras.ops.add(x1, x2)

x1 = keras.ops.convert_to_tensor(keras.random.normal(shape=(2, 3)))
x2 = keras.ops.convert_to_tensor(keras.random.normal(shape=(2, 3)))
example_output = call_func([x1, x2])