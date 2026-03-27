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

# Test with eager tensors (dynamic case)
print("Testing with eager tensors:")
query_eager = tf.random.normal((2, 4, 8, 16))
key_eager = tf.random.normal((2, 6, 8, 16))
value_eager = tf.random.normal((2, 6, 8, 16))
bias_eager = np.random.normal(size=(2, 1, 1, 6))
mask_eager = np.random.choice([True, False], size=(4, 1, 1, 6))

try:
    dynamic_output = call_func(
        inputs=[query_eager, key_eager, value_eager],
        bias=bias_eager,
        mask=mask_eager,
        scale=0.0,
        is_causal=False,
        flash_attention=False,
        attn_logits_soft_cap=None
    )
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic case error: {e}")

# Test with Keras Input placeholders (static case)
print("\nTesting with Keras Input placeholders:")
query_input = keras.Input(shape=(4, 8, 16))
key_input = keras.Input(shape=(6, 8, 16))
value_input = keras.Input(shape=(6, 8, 16))

try:
    static_output = call_func(
        inputs=[query_input, key_input, value_input],
        bias=bias_eager,
        mask=mask_eager,
        scale=0.0,
        is_causal=False,
        flash_attention=False,
        attn_logits_soft_cap=None
    )
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static case error: {e}")