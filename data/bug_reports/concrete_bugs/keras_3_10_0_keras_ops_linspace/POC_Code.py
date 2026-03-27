import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    start, stop = inputs[0], inputs[1]
    return keras.ops.linspace(start, stop, num, endpoint, retstep, dtype, axis)

# Test with eager tensors (dynamic)
eager_start = tf.constant(0.0)  # EagerTensor(shape=())
eager_stop = tf.constant(1.0)   # EagerTensor(shape=())
eager_inputs = [eager_start, eager_stop]

dynamic_output = call_func(eager_inputs, num=1, endpoint=True, retstep=False, dtype=None, axis=-1)
dynamic_shape = dynamic_output.shape

# Test with Keras.Input placeholders (static)
static_start = keras.Input(shape=(), dtype='float32')  # Keras.Input placeholder with shape=()
static_stop = keras.Input(shape=(), dtype='float32')   # Keras.Input placeholder with shape=()
static_inputs = [static_start, static_stop]

static_output = call_func(static_inputs, num=1, endpoint=True, retstep=False, dtype=None, axis=-1)
static_shape = static_output.shape

print(f"Dynamic output shape: {dynamic_shape}")
print(f"Static output shape: {static_shape}")
print(f"Shapes are consistent: {dynamic_shape == static_shape}")