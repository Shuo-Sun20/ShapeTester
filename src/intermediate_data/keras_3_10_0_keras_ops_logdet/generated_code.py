import keras
from keras import ops

def call_func(inputs):
    return ops.logdet(inputs)

n = 5
A = keras.random.normal(shape=(n, n))
B = ops.matmul(A, ops.transpose(A)) + n * ops.eye(n)

example_output = call_func(B)