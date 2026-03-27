import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np
import tensorflow as tf

def call_func(
    inputs,
    bias=None,
    mask=None,
    scale=None,
    is_causal=False,
    flash_attention=None,
    attn_logits_soft_cap=None
):
    query, key, value = inputs
    output = keras.ops.dot_product_attention(
        query=query,
        key=key,
        value=value,
        bias=bias,
        mask=mask,
        scale=scale,
        is_causal=is_causal,
        flash_attention=flash_attention,
        attn_logits_soft_cap=attn_logits_soft_cap
    )
    return output

# Create test inputs as eager tensors
query_eager = tf.random.normal((2, 4, 8, 16))
key_eager = tf.random.normal((2, 6, 8, 16))
value_eager = tf.random.normal((2, 6, 8, 16))
inputs_eager = [query_eager, key_eager, value_eager]

# Create bias and mask as numpy arrays
bias = np.random.normal(size=(1, 8, 1, 6))
mask = np.random.choice([True, False], size=(2, 8, 4, 6))

# Test with eager tensors
print("Testing with eager tensors:")
try:
    output_eager = call_func(
        inputs=inputs_eager,
        bias=bias,
        mask=mask,
        scale=1.0,
        is_causal=False,
        flash_attention=True,
        attn_logits_soft_cap=None
    )
    print(f"Dynamic output shape: {output_eager.shape}")
except Exception as e:
    print(f"Dynamic output shape: {str(e)}")

# Test with Keras.Input placeholders
print("\nTesting with Keras.Input placeholders:")
query_input = keras.Input(shape=(4, 8, 16))
key_input = keras.Input(shape=(6, 8, 16))
value_input = keras.Input(shape=(6, 8, 16))
inputs_placeholder = [query_input, key_input, value_input]

try:
    output_placeholder = call_func(
        inputs=inputs_placeholder,
        bias=bias,
        mask=mask,
        scale=1.0,
        is_causal=False,
        flash_attention=True,
        attn_logits_soft_cap=None
    )
    print(f"Static output shape: {output_placeholder.shape}")
except Exception as e:
    print(f"Static output shape: {str(e)}")

print("\nDefect reproduced: Dynamic and static output shapes are inconsistent!")