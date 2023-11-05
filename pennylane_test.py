import pennylane as qml
import numpy as np
import torch
import sys
import ast
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import paper as pa
import q_circuit as qc
import data_cake as cake

# -- USER INTERACTION --
# for which file with which hyperparameters the user is looking for, NO print_at as hyperparameter (only used in input!)
NUMBER_OF_EPOCHS = int(sys.argv[1])   # around 1000
NUMBER_OF_QUBITS = int(sys.argv[2])   # originally 5, test with more
NUMBER_OF_BLOCKS = int(sys.argv[3])   # originally 5, test with more
LEARNING_RATE = float(sys.argv[4])    # around 0.05 probably
NUMBER_OF_SECTORS = int(sys.argv[5])  # can be anything, originally 3
PLOTTING = int(sys.argv[6])           # 0 or 1

cake.NUMBER_OF_SECTORS = NUMBER_OF_SECTORS

# plotting = input("Do you wish to plot the decision boundaries? (longer waiting time) y/n")
#
# if plotting.lower() == "yes":
#     plotting = 1
# elif plotting.lower() == "no":
#     plotting = 0
# else:
#     print("Invalid response. Please enter 'yes' or 'no'.")


# Getting the trained parameters
parameters_string = []
test_parameter_file = "results/resulting_params" + "_" + str(NUMBER_OF_EPOCHS) + "_" + str(NUMBER_OF_QUBITS) + "_" + str(NUMBER_OF_BLOCKS) + \
                      "_" + str(LEARNING_RATE) + "_" + str(NUMBER_OF_SECTORS) + ".txt"
with open(test_parameter_file, 'r') as file:
    lines = file.readlines()
if len(lines) >= 2:
    parameters_string = lines[1]
else:
    print("Wrong file, please check again")
parameters_list = ast.literal_eval(parameters_string)


# Generation of random parameters depending on the given hyperparameters
torch.manual_seed(1401)  # same seed than in q_ann
def random_torch_params():
    return 2 * torch.pi * torch.rand((NUMBER_OF_BLOCKS, qc.NUMBER_OF_BLOCK_PARAMS, NUMBER_OF_QUBITS), dtype=torch.float64, requires_grad=False)


# Returns a float value which indicates how many datapoints were correctly classified
def get_accuracy(classifier, x, y):
    return 1 - np.count_nonzero(classifier.predict(x) - y) / len(y)


# Cake data initialization
(X, Y) = cake.data()
(X, Y) = (X.detach().numpy(), Y.detach().numpy())


# Tests the parameters with help of the pennylane library and also the kernel function of paper.py, needs to be adjusted
# in paper.py when testing a file with a different quantum embedding
def pennylane_tester(parameters, x, y):
    kernel = lambda x1, x2: pa.kernel(x1, x2, parameters)
    kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, kernel)
    kta = qml.kernels.target_alignment(x, y, kernel, assume_normalized_kernel=True)
    svm = SVC(kernel=kernel_matrix).fit(x, y)
    accuracy = get_accuracy(svm, x, y)
    print("accuracy:  ", accuracy)
    print("kta-value:  ", kta)
    return svm


# --- TRAINED VS. RANDOM PARAMETERS

print("-- UNTRAINED --")
random_parameters = random_torch_params().numpy()
svm_untrained = pennylane_tester(random_parameters, X, Y)
print("plotting the untrained decision boundaries...")
untrained_plot_data = cake.plot_decision_boundaries(svm_untrained, plt.gca(), X, Y)
print("close plot window to continue")
plt.show()


print("-- TRAINED --")
trained_parameters = np.array(parameters_list)
svm_trained = pennylane_tester(trained_parameters, X, Y)
if PLOTTING:
    print("plotting the trained decision boundaries...")
    trained_plot_data = cake.plot_decision_boundaries(svm_trained, plt.gca(), X, Y)
    print("done, close plot window to end the program")
    plt.show()


