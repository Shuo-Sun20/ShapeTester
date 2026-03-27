import numpy as np
import keras

def call_func(
    input_dim,
    output_dim,
    embeddings_initializer="uniform",
    embeddings_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    weights=None,
    lora_rank=None,
    lora_alpha=None,
    inputs=None
):
    embedding_layer = keras.layers.Embedding(
        input_dim=input_dim,
        output_dim=output_dim,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer,
        embeddings_constraint=embeddings_constraint,
        mask_zero=mask_zero,
        weights=weights,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha
    )
    output = embedding_layer(inputs)
    return output

np.random.seed(42)
input_tensor = np.random.randint(low=0, high=1000, size=(32, 10))
example_output = call_func(
    input_dim=1000,
    output_dim=64,
    inputs=input_tensor
)