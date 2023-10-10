import q_library as qt


# THIS FILE SERVES TO MODIFY THE QUANTUM CIRCUIT USING LAYERS FROM THE Q_LIBRARY

# THE CURRENT ONE IS FROM Thomas Hubregtsen, David Wierichs, Elies Gil-Fuster, Peter-Jan H. S. Derks, Paul K. Faehrmann, and Johannes Jakob Meyer.
# “Training Quantum Embedding Kernels on Near-Term Quantum Computers.” arXiv:2105.02276, 2021. page 10

# THE ANSATZ (BLOCK), THE ADJOINT ANSATZ (ADJ_BLOCK) & THE FINAL QUANTUM CIRCUIT
# Blocks composed by layers which then create the circuit
# The initial state is passed by the blocks and layers to get the resulting final state of the circuit

# Blocks:
# - start is the feature with which to start the RZ-layer

# Circuit:
# - j and k the number of blocks
# - For the adjoint ansatz: the block parameters taken last first but themselves in right order -> reversed(params)

# ---
def block(num_q, x, block_params, start, state):
    state = qt.h_layer(num_q, state)
    state = qt.rz_layer(num_q, x, state, start)
    state = qt.ry_layer(num_q, block_params[0], state)
    state = qt.crz_ring(num_q, block_params[1], state)
    return state


def adj_block(num_q, x, block_params, start, state):
    state = qt.adj_crz_ring(num_q, block_params[1], state)
    state = qt.adj_ry_layer(num_q, block_params[0], state)
    state = qt.adj_rz_layer(num_q, x, state, start)
    state = qt.h_layer(num_q, state)
    return state


def circuit(num_q, point1, point2, params):
    state = qt.initial_state(num_q)
    for j, block_params in enumerate(params):
        state = block(num_q, point1, block_params, j * num_q, state)
    for k, block_params in enumerate(reversed(params)):
        state = adj_block(num_q, point2, block_params, (num_q-k-1) * num_q, state)
    # print(state.grad_fn)
    return state
