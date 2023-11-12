import pennylane as qml

NUMBER_OF_WIRES = 5

# ADAPT THE CIRCUIT IN PENNYLANE
def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])


def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)

def init_circuit():
    dev = qml.device("default.qubit", wires=NUMBER_OF_WIRES, shots=None)
    wires = dev.wires.tolist()

    @qml.qnode(dev, interface="autograd")
    def kernel_circuit(x1, x2, params):
        ansatz(x1, params, wires=wires)
        adjoint_ansatz(x2, params, wires=wires)
        return qml.probs(wires=wires)

    return kernel_circuit

def kernel(x1, x2, params):
    return init_circuit()(x1, x2, params)[0]
