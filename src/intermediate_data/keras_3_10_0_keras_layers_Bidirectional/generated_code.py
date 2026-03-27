import keras
import keras.random

def call_func(layer, inputs, initial_state=None, mask=None, training=None, merge_mode="concat", weights=None, backward_layer=None):
    bidirectional_layer = keras.layers.Bidirectional(layer=layer, merge_mode=merge_mode, weights=weights, backward_layer=backward_layer)
    output = bidirectional_layer(inputs, initial_state=initial_state, mask=mask, training=training)
    return output

example_input = keras.random.normal(shape=(32, 10, 8))
example_layer = keras.layers.LSTM(16, return_sequences=True)
example_output = call_func(layer=example_layer, inputs=example_input)