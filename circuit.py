# import numpy as np
import torch
import torch.optim as optim
from torch import nn
# from sklearn.svm import SVC
import sys

import paper

import time
import cProfile
# from memory_profiler import profile

# ---
# NOTES

# - stagnates around 0.35, with learning rate 0.7
# - test with clusters
# - play with batch size, learning rate, other optimizers
# - could use more constants like DIM in CRZ and Close Ring but then less generalized code
#
# - positive semi-definite matrix?
# - use trained Kernel as Classification
# ---


# ---
# SETTINGS
# ---
torch.set_printoptions(precision=10, sci_mode=False)    # Set precision to 8 decimal places to fit with paper
torch.manual_seed(1401)
start_time = time.time()

# device = torch.device("cpu")


# ---
# DATESET
# From the paper implementation paper.py
# ---
num_sectors = paper.NUMBER_OF_SECTORS
(X, Y) = paper.make_double_cake_data(num_sectors)
ax = paper.plot_double_cake_data(X, Y, paper.plt.gca(), num_sectors)
(X, Y) = (torch.tensor(X, dtype=torch.float64, requires_grad=True), torch.tensor(Y, dtype=torch.float64, requires_grad=True))
# paper.plt.show()


# ---
# DEFINITION OF MATRICES
# Defining the gates for a single qubit i.e. the most simple form
# - stack instead of tensor to make forward differentiable
# - around 3 million calls for 100 epochs
# ---

sqrt_of_two = torch.sqrt(torch.tensor(2, dtype=torch.complex128))   # saving computation by pre calculating
hadamard_single = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / sqrt_of_two


def ry_matrix(angle):
    cos_half_angle = torch.cos(angle*0.5)
    sin_half_angle = torch.sin(angle*0.5)

    line1 = torch.stack((cos_half_angle, -sin_half_angle))
    line2 = torch.stack((sin_half_angle, cos_half_angle))

    ry_mat = torch.stack((line1, line2)).type(torch.complex128)

    return ry_mat

def ry_matrix_t(angle):
    return torch.conj(ry_matrix(angle)).t()


def rz_matrix(angle):
    neg_exp = torch.exp(-0.5j*angle)
    pos_exp = torch.exp(0.5j*angle)
    zero_int = torch.tensor(0)

    line1 = torch.stack((neg_exp, zero_int))
    line2 = torch.stack((zero_int, pos_exp))

    rz_mat = torch.stack((line1, line2)).type(torch.complex128)

    return rz_mat

def rz_matrix_t(angle):
    return torch.conj(rz_matrix(angle)).t()


# The Controlled RZ Matrix
# returns the matrix for rz on certain qubit q and a control on q - 1
# CONDITION: q - c must be one. Differentiable Function
def crz_matrix(q, num_q, angle):
    dim = 2 ** num_q
    per_length = 2 ** q * 2  # how long is an interval
    num_rep = 2 ** (q - 1)  # how often is a value repeated
    help_diag = torch.ones(per_length, dtype=torch.complex128)
    for k in range(num_rep):
        help_diag[num_rep + k] = rz_matrix(angle)[0][0]
        help_diag[num_rep + k + per_length // 2] = rz_matrix(angle)[1][1]
    final_diag = help_diag
    # join the intervals in dimension length
    for _ in range(dim // per_length - 1):
        final_diag = torch.cat((final_diag, help_diag))
    # crz = torch.from_numpy(final_diag).type(torch.complex64)
    crz = torch.diag(final_diag)
    return crz

def close_ring(num_q, angle):
    dim = 2 ** num_q
    values = torch.tensor([rz_matrix(angle)[0][0], rz_matrix(angle)[1][1]], requires_grad=True)
    help_diag = torch.ones(dim // 2, dtype=torch.complex128, requires_grad=True)  # first half only ones for control
    help_diag = torch.cat((help_diag, values.tile((dim // 4,))))  # The second half is alternately filled with - +
    close_matrix = torch.diag(help_diag)
    return close_matrix


# ---
# DEFINITION OF LAYERS
#  composing of the simple matrices to build same-type-gate parallel layers or rings in the size of the given number of quantum's of the circuit
#  adjacent function always apart since conditions in function break the computation graph

# for the rz layers: In case len(angles) > qubits (number of features > number of qubits) the order is for ansatz: 1, 6, 4, 2, 7, 5; adjoint: 5, 7, 2, 4, 6, 1
# ---

def crz_ring(num_q, angles, state):
    for i in range(num_q - 1):
        state = torch.matmul(crz_matrix(i + 1, num_q, angles[i]), state)
    state = torch.matmul(close_ring(num_q, angles[-1]), state)
    return state

def adj_crz_ring(num_q, angles, state):
    state = torch.matmul(torch.conj(close_ring(num_q, angles[-1]).t()), state)
    for i in range(num_q - 1, 0, -1):  # backwards
        state = torch.matmul(torch.conj(crz_matrix(i, num_q, angles[i-1])).t(), state)
    return state


def ry_layer(num_q, angles, state):
    ry = ry_matrix(angles[-1])
    for i in range(num_q - 1):
        ry = torch.kron(ry, ry_matrix(angles[(num_q - 1 - (i+1))]))
    state = torch.matmul(ry, state)
    return state

def adj_ry_layer(num_q, angles, state):
    ry = ry_matrix_t(angles[-1])
    for i in range(num_q - 1):
        ry = torch.kron(ry, ry_matrix_t(angles[(num_q - 1 - (i+1))]))
    state = torch.matmul(ry, state)
    return state


def rz_layer(num_q, angles, state, start):
    rz = rz_matrix(angles[start % len(angles)])
    for _ in range(num_q - 1):
        start += 1
        rz = torch.kron(rz, rz_matrix(angles[start % len(angles)]))
    state = torch.matmul(rz, state)
    return state

def adj_rz_layer(num_q, angles, state, start):
    rz = rz_matrix_t(angles[start % len(angles)])
    for _ in range(num_q - 1):
        start += 1
        rz = torch.kron(rz, rz_matrix_t(angles[start % len(angles)]))
    state = torch.matmul(rz, state)
    return state


def h_layer(num_q, state):
    h = hadamard_single
    for _ in range(num_q - 1):
        h = torch.kron(h, hadamard_single)
    return torch.matmul(h, state)


# ---
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
    state = h_layer(num_q, state)
    state = rz_layer(num_q, x, state, start)
    state = ry_layer(num_q, block_params[0], state)
    state = crz_ring(num_q, block_params[1], state)
    return state

def adj_block(num_q, x, block_params, start, state):
    state = adj_crz_ring(num_q, block_params[1], state)
    state = adj_ry_layer(num_q, block_params[0], state)
    state = adj_rz_layer(num_q, x, state, start)
    state = h_layer(num_q, state)
    return state


def circuit(num_q, point1, point2, params):
    state = initial_state(num_q)
    for j, block_params in enumerate(params):
        state = block(num_q, point1, block_params, j * num_q, state)
    for k, block_params in enumerate(reversed(params)):
        state = adj_block(num_q, point2, block_params, (num_q-k-1) * num_q, state)
    # print(state.grad_fn)
    return state


# ---
# HELPFUL FUNCTIONS
# probability:      Converts a state to the resulting probability. Probability of the state to be in the zero state |0...0⟩
# initial_state:    Define the initial state as |0...0⟩
# random_params:    Generate random variational parameters in the shape for the ansatz respectively for adjoint ansatz
# ---

def probability(state):
    prob = torch.abs(state) ** 2
    return prob

def initial_state(n_qubits):
    return torch.tensor([1] + [0] * (2 ** n_qubits - 1), dtype=torch.complex128, requires_grad=True)

def random_params(n_blocks, n_qubits):
    return 2 * torch.pi * torch.rand((n_blocks, 2, n_qubits), dtype=torch.float64, requires_grad=True)

def paper_params():
    return torch.from_numpy(paper.init_params)


# ---
# KERNEL FUNCTIONS
#   # Later pass the circuit as a function with the inputs to make program general
#
# -
# KERNEL-TARGET-ALIGNMENT
# page 6 in paper, works fine, is differentiable
# used as loss function in the artificial neural net
#
# ---

#  original one, not used, k_fusion instead
def kernel_matrix(dataset, num_q, params):
    input_length = len(dataset)
    kernel_mat = torch.empty(input_length, input_length, dtype=torch.float64)
    for i in range(input_length):
        for j in range(input_length):
            kernel_mat[i][j] = probability(circuit(num_q, dataset[i], dataset[j], params)[0])
    # print(kernel_mat.grad_fn)       # Does not work... different way to build the matrix?
    return kernel_mat


# not used in current program -> k_fusion
def kernel_target_alignment(k, labels):
    kernel_polarity = torch.sum(labels.view(-1, 1) * labels.view(1, -1) * k)
    square_sum_k = torch.sum(k ** 2)
    square_sum_l = torch.sum((labels.view(-1, 1) * labels.view(1, -1)) ** 2)
    normalization = torch.sqrt(square_sum_k*square_sum_l)
    # print(kernel_polarity, normalization)
    ta_al = kernel_polarity / normalization
    return ta_al


# Instead of building the kernel matrix which interrupts the backward graph, uses the computed values directly to calculate the kta-value
def k_fusion(dataset, num_q, params, labels):
    # to optimize computation, dividing in the forward instead of in the matrices, doesn't really change much
    # dataset = dataset * 0.5
    # params = params * 0.5
    num_points = len(labels)
    kernel_polarity = 0
    square_sum_k = 0
    square_sum_l = 0
    for i in range(num_points):
        for j in range(num_points):
            k = probability(circuit(num_q, dataset[i], dataset[j], params)[0])
            kernel_polarity = labels[i]*labels[j]*k + kernel_polarity
            square_sum_k = k ** 2 + square_sum_k
            square_sum_l = (labels[i]*labels[j]) ** 2 + square_sum_l
    normalization = torch.sqrt(square_sum_k*square_sum_l)
    fusion_val = kernel_polarity / normalization
    return fusion_val


# ---
# INITIALISATION OF VARIABLES
#
# epochs:       How often optimized in training process
# print_at:     Every "print_at" times the loss is printed
#
# qubits/wires: 5 used in paper, too high number of qubits (10) -> DefaultCPUAllocator: not enough memory
# block number: (how often is circuit repeated): "issue" only impair numbers work correctly by now
#
# ---

NUMBER_OF_EPOCHS = int(sys.argv[1])  # 1000
PRINT_AT = int(sys.argv[2])          # 100

NUMBER_OF_QUBITS = int(sys.argv[3])  # 5
NUMBER_OF_BLOCKS = int(sys.argv[4])  # 5
LEARNING_RATE = int(sys.argv[5])     # 0.05


# ---
# The QUANTUM ARTIFICIAL NEURAL NETWORK class and its initialization
#
# Interprets a given Quantum Circuit as a Neural Net. The parameters of the circuit take the role of the weights.
# For the forward method the k_fusion function is used which returns directly the kernel-target-alignment-value
# ("the measure of how well the kernel captures the nature of the training dataset"-page 5 in paper),
# therefore the labels of the given dataset is needed
#
# ---
class QuantumNet(nn.Module):
    def __init__(self, n_qubits, parameters):
        super(QuantumNet, self).__init__()
        self.n_qubits = n_qubits
        self.params = nn.Parameter(parameters)  # weights
        self.fusion = k_fusion

    def forward(self, data, labels):
        kta = self.fusion(data, self.n_qubits, self.params, labels)
        return kta


# ---
# LOSS FUNCTIONS
# loss2      is in case the kernel matrix is returned from the net and calculates therefore the kta.
#            Not used, k_fusion computes kta directly
#
# kta_loss   expects the kta value (which needs to be maximized and is in [0,1]) and shifts the sign to negative, so it can be minimized as a loss function
#
# ---

def loss2(ker_mat, labels):
    loss_1 = -kernel_target_alignment(ker_mat, labels)
    return loss_1

def kta_loss(kta_val):
    return 0-kta_val


# ---
# TRAINING:
#
# Training Function used to keep track of time and memory usage
#
# ---
model = QuantumNet(n_qubits=NUMBER_OF_QUBITS, parameters=random_params(NUMBER_OF_BLOCKS, NUMBER_OF_QUBITS))

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train()

def train(n_epochs, print_at):
    for epoch in range(n_epochs):

        optimizer.zero_grad()

        # subset = torch.randperm(X.size(0))[:3]

        kta_value = model(X, Y)

        loss = kta_loss(kta_value)

        loss.backward()

        optimizer.step()

        if epoch % print_at == 0:
            print("epoch:", epoch, 'loss:', -loss.item())


cProfile.run("train(NUMBER_OF_EPOCHS, PRINT_AT)")


end_time = time.time()
exec_time = end_time - start_time
print(f"Execution Time: {exec_time}")

