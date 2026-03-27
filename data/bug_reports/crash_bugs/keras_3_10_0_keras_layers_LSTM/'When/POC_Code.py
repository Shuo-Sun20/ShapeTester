import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

def call_func(
    units,
    activation="tanh",
    recurrent_activation="sigmoid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    seed=None,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    use_cudnn="auto",
    inputs=None,
    mask=None,
    training=None,
    initial_state=None
):
    lstm_layer = keras.layers.LSTM(
        units=units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        seed=seed,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        use_cudnn=use_cudnn
    )
    
    if initial_state is not None:
        output = lstm_layer(inputs, mask=mask, training=training, initial_state=initial_state)
    else:
        output = lstm_layer(inputs, mask=mask, training=training)
    
    return output

# Test parameters from the defect input
test_params = {
    'units': 1,
    'activation': 'tanh',
    'recurrent_activation': 'sigmoid',
    'use_bias': True,
    'kernel_initializer': 'glorot_uniform',
    'recurrent_initializer': 'orthogonal',
    'bias_initializer': 'zeros',
    'unit_forget_bias': True,
    'kernel_regularizer': None,
    'recurrent_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'recurrent_constraint': None,
    'bias_constraint': None,
    'dropout': 0.0,
    'recurrent_dropout': 0.0,
    'seed': None,
    'return_sequences': True,
    'return_state': True,
    'go_backwards': True,
    'stateful': True,
    'unroll': True,
    'use_cudnn': 'auto',
    'mask': None,
    'training': None,
    'initial_state': None
}

# Test with eager tensors (dynamic)
print("Testing with eager tensors:")
eager_input = tf.constant(np.random.random((32, 10, 8)), dtype=tf.float32)
test_params['inputs'] = eager_input

try:
    dynamic_output = call_func(**test_params)
    if isinstance(dynamic_output, tuple):
        print("Dynamic output shapes:", [list(out.shape) for out in dynamic_output])
    else:
        print("Dynamic output shape:", list(dynamic_output.shape))
except Exception as e:
    print("Dynamic execution error:", str(e))

# Test with Keras.Input placeholders (static)
print("\nTesting with Keras.Input placeholders:")
static_input = keras.Input(shape=(10, 8), batch_size=32)  # Fixed batch size for stateful
test_params['inputs'] = static_input

try:
    static_output = call_func(**test_params)
    if isinstance(static_output, tuple):
        print("Static output shapes:", [list(out.shape) for out in static_output])
    else:
        print("Static output shape:", list(static_output.shape))
except Exception as e:
    print("Static execution error:", str(e))

# Test with Keras.Input without fixed batch size to trigger the error
print("\nTesting with Keras.Input without fixed batch size:")
static_input_no_batch = keras.Input(shape=(10, 8))
test_params['inputs'] = static_input_no_batch

try:
    static_output_no_batch = call_func(**test_params)
    if isinstance(static_output_no_batch, tuple):
        print("Static output shapes (no batch):", [list(out.shape) for out in static_output_no_batch])
    else:
        print("Static output shape (no batch):", list(static_output_no_batch.shape))
except Exception as e:
    print("Static execution error (no batch):", str(e))