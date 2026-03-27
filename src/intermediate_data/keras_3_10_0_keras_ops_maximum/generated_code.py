import keras

def call_func(inputs):
    return keras.ops.maximum(inputs[0], inputs[1])

tensor1 = keras.random.normal(shape=(3, 4))
tensor2 = keras.random.normal(shape=(3, 4))
example_output = call_func([tensor1, tensor2])