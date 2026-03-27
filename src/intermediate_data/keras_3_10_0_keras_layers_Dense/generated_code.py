import keras
import numpy as np

def call_func(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    lora_rank=None,
    lora_alpha=None
):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    
    if lora_rank is not None:
        kernel_constraint = None
    
    dense_layer = keras.layers.Dense(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha
    )
    
    output = dense_layer(input_tensor)
    return output

input_tensor = np.random.randn(4, 8).astype(np.float32)
example_output = call_func(
    inputs=input_tensor,
    units=16,
    activation="relu",
    use_bias=True,
    lora_rank=4
)