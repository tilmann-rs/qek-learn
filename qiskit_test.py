import qiskit
from qiskit import QuantumCircuit, transpile, assemble, execute, Aer

import data_cake as cake
import q_ann

def block(circuit, x, block_params, wires, start):
    """Building block of the embedding ansatz"""

    for j, wire in enumerate(wires):
        circuit.h(wire)
        circuit.rz(x[start % len(x)], wire)
        start += start
        circuit.ry(block_params[0, j], wire)

    for i in range(len(wires) - 1):
        qubit1 = wires[i]
        qubit2 = wires[i + 1]
        circuit.crz(block_params[1][i], qubit1, qubit2)


num_qubits = 5
num_blocks = 1
dataset = np.random.rand(num_qubits * num_blocks)
parameters = np.random.rand(num_blocks, 2, num_qubits)
wires = list(range(num_qubits))

def ansatz(x, circ, params, wres):
    for j, block_params in enumerate(params):
        block(circ, x, block_params, wres, j * len(wres))

    circ = circ.compose(circ.inverse(), inplace=True)
    return circ

def test_in_qiskit(number_qubits, number_blocks, trained_params):

    (X, Y) = cake.data()

    circuit = QuantumCircuit(num_qubits)


    k_val = q_ann.k_fusion(X, num_qubits, trained_params, Y)

    return circuit


# Simulate the circuit using Aer's QASM simulator
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(circuit, simulator)
job = execute(compiled_circuit, simulator, shots=1024)
result = job.result()
counts = result.get_counts(circuit)
print(counts)
