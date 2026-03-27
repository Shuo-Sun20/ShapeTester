import torch

def call_func(inputs, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """
    Calls torch.functional.F.scaled_dot_product_attention with provided parameters.
    
    Args:
        inputs: List containing three tensors [query, key, value]
        attn_mask: Optional attention mask tensor
        dropout_p: Dropout probability (default: 0.0)
        is_causal: Whether to apply causal masking (default: False)
        scale: Scaling factor (default: None)
        enable_gqa: Enable Grouped Query Attention (default: False)
        
    Returns:
        Output tensor from the attention operation
    """
    query, key, value = inputs
    
    # Call the API directly
    return torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa
    )

# Construct a valid input
batch_size = 2
num_heads_query = 8
num_heads_kv = 4  # For demonstrating different head counts
seq_len_q = 10
seq_len_kv = 12
embed_dim = 64
value_dim = 48

# Create random tensors
query = torch.randn(batch_size, num_heads_query, seq_len_q, embed_dim)
key = torch.randn(batch_size, num_heads_kv, seq_len_kv, embed_dim)
value = torch.randn(batch_size, num_heads_kv, seq_len_kv, value_dim)

# Call the function
example_output = call_func(
    inputs=[query, key, value],
    attn_mask=None,
    dropout_p=0.1,
    is_causal=True,
    scale=None,
    enable_gqa=True
)