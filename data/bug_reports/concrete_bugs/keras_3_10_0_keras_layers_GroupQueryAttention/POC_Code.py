import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np
import tensorflow as tf

def call_func(
    head_dim,
    num_query_heads,
    num_key_value_heads,
    inputs,
    dropout=0.0,
    use_bias=True,
    flash_attention=None,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    seed=None,
    attention_mask=None,
    return_attention_scores=False,
    training=None,
    use_causal_mask=False
):
    layer = keras.layers.GroupQueryAttention(
        head_dim=head_dim,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        dropout=dropout,
        use_bias=use_bias,
        flash_attention=flash_attention,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        seed=seed
    )
    
    if len(inputs) == 2:
        query, value = inputs
        key = None
    elif len(inputs) == 3:
        query, value, key = inputs
    else:
        raise ValueError("Inputs must contain 2 or 3 tensors")
    
    output = layer(
        query=query,
        value=value,
        key=key,
        attention_mask=attention_mask,
        return_attention_scores=return_attention_scores,
        training=training,
        use_causal_mask=use_causal_mask
    )
    
    if return_attention_scores:
        return output[0]
    return output

# Test parameters
head_dim = 64
num_query_heads = 2
num_key_value_heads = 2
dropout = 0.1
use_bias = True
flash_attention = None
kernel_initializer = "glorot_uniform"
bias_initializer = "zeros"
kernel_regularizer = None
bias_regularizer = None
activity_regularizer = None
kernel_constraint = None
bias_constraint = None
seed = None
attention_mask = None
return_attention_scores = True
training = None
use_causal_mask = False

# Create eager tensors
query_eager = tf.constant(np.random.randn(2, 10, 512), dtype=tf.float32)
value_eager = tf.constant(np.random.randn(2, 15, 512), dtype=tf.float32)
key_eager = tf.constant(np.random.randn(2, 15, 512), dtype=tf.float32)
inputs_eager = [query_eager, value_eager, key_eager]

# Create Keras Input placeholders
query_placeholder = keras.Input(shape=(10, 512))
value_placeholder = keras.Input(shape=(15, 512))
key_placeholder = keras.Input(shape=(15, 512))
inputs_placeholder = [query_placeholder, value_placeholder, key_placeholder]

# Call with eager tensors
output_eager = call_func(
    head_dim=head_dim,
    num_query_heads=num_query_heads,
    num_key_value_heads=num_key_value_heads,
    inputs=inputs_eager,
    dropout=dropout,
    use_bias=use_bias,
    flash_attention=flash_attention,
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer,
    kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer,
    activity_regularizer=activity_regularizer,
    kernel_constraint=kernel_constraint,
    bias_constraint=bias_constraint,
    seed=seed,
    attention_mask=attention_mask,
    return_attention_scores=return_attention_scores,
    training=training,
    use_causal_mask=use_causal_mask
)

# Call with Keras Input placeholders
output_placeholder = call_func(
    head_dim=head_dim,
    num_query_heads=num_query_heads,
    num_key_value_heads=num_key_value_heads,
    inputs=inputs_placeholder,
    dropout=dropout,
    use_bias=use_bias,
    flash_attention=flash_attention,
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer,
    kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer,
    activity_regularizer=activity_regularizer,
    kernel_constraint=kernel_constraint,
    bias_constraint=bias_constraint,
    seed=seed,
    attention_mask=attention_mask,
    return_attention_scores=return_attention_scores,
    training=training,
    use_causal_mask=use_causal_mask
)

# Print shapes to demonstrate the defect
print(f"Dynamic output shape (eager tensors): {output_eager.shape}")
print(f"Static output shape (placeholders): {output_placeholder.shape}")
print(f"Shape mismatch detected: {output_eager.shape != output_placeholder.shape}")