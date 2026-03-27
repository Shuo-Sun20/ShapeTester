import numpy as np
import keras

def call_func(cells, inputs, training=False):
    stacked_cell = keras.layers.StackedRNNCells(cells)
    # The states should be a list of state tensors for each cell
    # For LSTMCell, each cell needs a list of two states: [h, c]
    states = []
    idx = 1  # Start from 1 because inputs[0] is the input tensor
    for cell in cells:
        if isinstance(cell, keras.layers.LSTMCell):
            # Each LSTM cell needs two state tensors
            h_state = inputs[idx]
            c_state = inputs[idx + 1]
            states.append([h_state, c_state])
            idx += 2
        else:
            # For other cell types that use single state tensor
            state = inputs[idx]
            states.append(state)
            idx += 1
    
    # Call the stacked cell with input tensor and initial states
    output = stacked_cell(inputs[0], states, training=training)
    return output[0]  # Return only the output, not the states

# Generate input data for example
batch_size = 3
features = 5
units = 128

# Input tensor for one timestep (batch_size, features)
input_tensor = np.random.randn(batch_size, features).astype(np.float32)

# Create two LSTM cells
rnn_cells = [keras.layers.LSTMCell(units) for _ in range(2)]

# Initial states for each LSTM cell
# Each LSTM cell needs [h_state, c_state]
h_state_1 = np.random.randn(batch_size, units).astype(np.float32)
c_state_1 = np.random.randn(batch_size, units).astype(np.float32)
h_state_2 = np.random.randn(batch_size, units).astype(np.float32)
c_state_2 = np.random.randn(batch_size, units).astype(np.float32)

# Combine all inputs into a single list
inputs_list = [input_tensor, h_state_1, c_state_1, h_state_2, c_state_2]

# Call the function
example_output = call_func(rnn_cells, inputs_list)