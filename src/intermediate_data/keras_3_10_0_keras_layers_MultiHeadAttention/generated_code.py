import keras
import numpy as np

def call_func(
    inputs,
    num_heads,
    key_dim,
    value_dim=None,
    dropout=0.0,
    use_bias=True,
    output_shape=None,
    attention_axes=None,
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
    training=False,
    use_causal_mask=False
):
    if len(inputs) == 3:
        query, key, value = inputs
    elif len(inputs) == 2:
        query, value = inputs
        key = None
    else:
        raise ValueError("Inputs must be list of 2 or 3 tensors")
    
    mha_layer = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        dropout=dropout,
        use_bias=use_bias,
        output_shape=output_shape,
        attention_axes=attention_axes,
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
    
    output = mha_layer(
        query=query,
        value=value,
        key=key,
        attention_mask=attention_mask,
        return_attention_scores=return_attention_scores,
        training=training,
        use_causal_mask=use_causal_mask
    )
    
    return output

batch_size = 2
seq_len = 5
feature_dim = 8
query_tensor = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)
value_tensor = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)
key_tensor = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)

example_output = call_func(
    inputs=[query_tensor, value_tensor, key_tensor],
    num_heads=2,
    key_dim=4,
    value_dim=4,
    dropout=0.1,
    use_bias=True,
    attention_mask=None,
    return_attention_scores=False,
    training=False,
    use_causal_mask=False
)