import torch
import torch.optim as optim
from torch import nn

import data_cake as cake
import q_circuit as qc

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


# ---
# HELPFUL FUNCTIONS
# probability:      Converts a state to the resulting probability. Probability of the state to be in the zero state |0...0⟩
# random_params:    Generate random variational parameters in the shape for the ansatz respectively for adjoint ansatz
# ---

def probability(state):
    prob = torch.abs(state) ** 2
    return prob

# def paper_params():
#     return torch.from_numpy(paper.init_params)

def random_params(n_blocks, n_qubits):
    return 2 * torch.pi * torch.rand((n_blocks, 2, n_qubits), dtype=torch.float64, requires_grad=True)


# ---
# KERNEL FUNCTIONS
# Later pass the circuit as a function with the inputs to make program general
# not used, k_fusion instead

# KERNEL-TARGET-ALIGNMENT
# page 6 in paper, works fine, is differentiable
# used as loss function in the artificial neural net
#
# K-FUSION
# Instead of building the kernel matrix which interrupts the backward graph, uses the computed values directly to calculate the kta-value
#
# ---

def kernel_matrix(dataset, num_q, params):
    input_length = len(dataset)
    kernel_mat = torch.empty(input_length, input_length, dtype=torch.float64)
    for i in range(input_length):
        for j in range(input_length):
            kernel_mat[i][j] = probability(qc.circuit(num_q, dataset[i], dataset[j], params)[0])
    # print(kernel_mat.grad_fn)       # Does not work... different way to build the matrix?
    return kernel_mat


def kernel_target_alignment(k, labels):
    kernel_polarity = torch.sum(labels.view(-1, 1) * labels.view(1, -1) * k)
    square_sum_k = torch.sum(k ** 2)
    square_sum_l = torch.sum((labels.view(-1, 1) * labels.view(1, -1)) ** 2)
    normalization = torch.sqrt(square_sum_k*square_sum_l)
    # print(kernel_polarity, normalization)
    ta_al = kernel_polarity / normalization
    return ta_al


def k_fusion(dataset, num_q, params, labels):
    num_points = len(labels)
    kernel_polarity = 0
    square_sum_k = 0
    square_sum_l = 0
    for i in range(num_points):
        for j in range(num_points):
            k = probability(qc.circuit(num_q, dataset[i], dataset[j], params)[0])
            kernel_polarity = labels[i]*labels[j]*k + kernel_polarity
            square_sum_k = k ** 2 + square_sum_k
            square_sum_l = (labels[i]*labels[j]) ** 2 + square_sum_l
    normalization = torch.sqrt(square_sum_k*square_sum_l)
    fusion_val = kernel_polarity / normalization
    return fusion_val


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
# INSTANTIATE & TRAINING QUANTUM MODEL (FUNCTION)
#   - saves the resulting optimized parameters in separate file
#
# ---
def train(n_epochs, print_at, n_qubits, n_blocks, learning_rate):

    model = QuantumNet(n_qubits=n_qubits, parameters=random_params(n_blocks, n_qubits))

    (X, Y) = cake.data()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(n_epochs):

        optimizer.zero_grad()

        # subset = torch.randperm(X.size(0))[:3]

        kta_value = model(X, Y)

        loss = kta_loss(kta_value)

        loss.backward()

        optimizer.step()

        if epoch % print_at == 0:
            print("epoch:", epoch, 'kta-value:', -loss.item())

    result = model.state_dict().values()
    trained_params = [tensor.tolist() for tensor in result][0]

    return trained_params

    # trained_model = QuantumNet(n_qubits=n_qubits, parameters=result)
    # result_kta = trained_model(X, Y)
    # print('kta-value:', result_kta.item())

