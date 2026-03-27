import torch
import torch.nn.functional as F

def call_func(
    inputs,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k=None,
    bias_v=None,
    add_zero_attn=False,
    dropout_p=0.0,
    out_proj_weight=None,
    out_proj_bias=None,
    training=True,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    is_causal=False,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None,
    average_attn_weights=True
):
    query, key, value = inputs
    attn_output, attn_output_weights = F.multi_head_attention_forward(
        query=query,
        key=key,
        value=value,
        embed_dim_to_check=embed_dim_to_check,
        num_heads=num_heads,
        in_proj_weight=in_proj_weight,
        in_proj_bias=in_proj_bias,
        bias_k=bias_k,
        bias_v=bias_v,
        add_zero_attn=add_zero_attn,
        dropout_p=dropout_p,
        out_proj_weight=out_proj_weight,
        out_proj_bias=out_proj_bias,
        training=training,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
        is_causal=is_causal,
        use_separate_proj_weight=use_separate_proj_weight,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        static_k=static_k,
        static_v=static_v,
        average_attn_weights=average_attn_weights
    )
    return attn_output

# Generate random input tensors
L, S, N, E = 3, 4, 2, 8
num_heads = 2

query = torch.randn(L, N, E)
key = torch.randn(S, N, E)
value = torch.randn(S, N, E)
inputs = [query, key, value]

in_proj_weight = torch.randn(3 * E, E)
in_proj_bias = torch.randn(3 * E)
out_proj_weight = torch.randn(E, E)
out_proj_bias = torch.randn(E)

example_output = call_func(
    inputs=inputs,
    embed_dim_to_check=E,
    num_heads=num_heads,
    in_proj_weight=in_proj_weight,
    in_proj_bias=in_proj_bias,
    out_proj_weight=out_proj_weight,
    out_proj_bias=out_proj_bias,
    need_weights=False
)