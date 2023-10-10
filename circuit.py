# import numpy as np
import torch
import torch.optim as optim
from torch import nn
# from sklearn.svm import SVC
import sys

import paper
import q_library as qt

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
LEARNING_RATE = float(sys.argv[5])     # 0.05


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

