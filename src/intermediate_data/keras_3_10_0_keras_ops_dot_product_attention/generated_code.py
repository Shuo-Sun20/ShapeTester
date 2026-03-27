import keras

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

query = keras.random.normal((2, 4, 8, 16))
key = keras.random.normal((2, 6, 8, 16))
value = keras.random.normal((2, 6, 8, 16))
example_output = call_func(inputs=[query, key, value])