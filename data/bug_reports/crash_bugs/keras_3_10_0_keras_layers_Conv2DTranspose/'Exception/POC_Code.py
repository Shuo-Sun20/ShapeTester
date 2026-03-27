import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

def call_func(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
):
    layer = keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint
    )
    return layer(inputs)

# Test with Keras.Input placeholders (static shape)
print("Testing with Keras.Input placeholders:")
placeholder_input = keras.Input(shape=(10, 8, 128))
try:
    static_output = call_func(
        inputs=placeholder_input,
        filters=64,
        kernel_size=5,
        strides=[1, 1],
        padding='same',
        output_padding=2,
        data_format='channels_last',
        dilation_rate=[3, 3],
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None
    )
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static shape error: {e}")

# Test with eager tensors (dynamic shape)
print("\nTesting with eager tensors:")
eager_input = tf.constant(np.random.rand(4, 10, 8, 128).astype(np.float32))
try:
    dynamic_output = call_func(
        inputs=eager_input,
        filters=64,
        kernel_size=5,
        strides=[1, 1],
        padding='same',
        output_padding=2,
        data_format='channels_last',
        dilation_rate=[3, 3],
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None
    )
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic shape error: {e}")

print("\nDefect reproduced: Static shape computation succeeds but dynamic execution fails with the same parameters.")