import keras

def call_func(inputs):
    x1, x2 = inputs
    return keras.ops.dot(x1, x2)

tensor1 = keras.random.normal(shape=(3, 4))
tensor2 = keras.random.normal(shape=(4, 5))
example_output = call_func([tensor1, tensor2])