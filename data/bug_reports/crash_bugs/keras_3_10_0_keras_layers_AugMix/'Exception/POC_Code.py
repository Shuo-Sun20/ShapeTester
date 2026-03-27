import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
from keras.layers import AugMix

def call_func(
    value_range=(0, 255),
    num_chains=3,
    chain_depth=3,
    factor=0.3,
    alpha=1.0,
    all_ops=True,
    interpolation="bilinear",
    seed=None,
    data_format=None,
    inputs=None
):
    augmix_instance = AugMix(
        value_range=value_range,
        num_chains=num_chains,
        chain_depth=chain_depth,
        factor=factor,
        alpha=alpha,
        all_ops=all_ops,
        interpolation=interpolation,
        seed=seed,
        data_format=data_format
    )
    if isinstance(inputs, list):
        output = augmix_instance(inputs[0])
    else:
        output = augmix_instance(inputs)
    return output

# Test with eager tensor (causes defect)
print("Testing with eager tensor:")
eager_tensor = tf.random.uniform((2, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)
try:
    dynamic_output = call_func(
        value_range=(0, 255),
        num_chains=3,
        chain_depth=3,
        factor=0.3,
        alpha=1.0,
        all_ops=True,
        interpolation="bilinear",
        seed=None,
        data_format="channels_first",
        inputs=eager_tensor
    )
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception encountered when calling AugMix.call().")
    print(f"{e}")

# Test with Keras Input placeholder (works correctly)
print("\nTesting with Keras Input placeholder:")
input_placeholder = keras.Input(shape=(224, 224, 3))
static_output = call_func(
    value_range=(0, 255),
    num_chains=3,
    chain_depth=3,
    factor=0.3,
    alpha=1.0,
    all_ops=True,
    interpolation="bilinear",
    seed=None,
    data_format="channels_first",
    inputs=input_placeholder
)
print(f"Static output shape: {static_output.shape}")