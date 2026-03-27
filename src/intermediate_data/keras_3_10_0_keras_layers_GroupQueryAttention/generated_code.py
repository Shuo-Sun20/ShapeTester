import keras
import numpy as np

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

# Generate random input tensors
batch_size = 2
target_seq_len = 10
source_seq_len = 15
head_dim = 64
num_query_heads = 8
num_key_value_heads = 2

query_feature_dim = head_dim * num_query_heads
kv_feature_dim = head_dim * num_key_value_heads

np.random.seed(42)
query = np.random.randn(batch_size, target_seq_len, query_feature_dim).astype(np.float32)
value = np.random.randn(batch_size, source_seq_len, kv_feature_dim).astype(np.float32)
key = np.random.randn(batch_size, source_seq_len, kv_feature_dim).astype(np.float32)

# Call the function
example_output = call_func(
    head_dim=head_dim,
    num_query_heads=num_query_heads,
    num_key_value_heads=num_key_value_heads,
    inputs=[query, value, key],
    dropout=0.1,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    attention_mask=None,
    return_attention_scores=False,
    use_causal_mask=False
)