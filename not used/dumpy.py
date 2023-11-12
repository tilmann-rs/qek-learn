
# trained_parameters = np.array(parameters_list)
# trained_kernel = lambda x1, x2: pa.kernel(x1, x2, trained_parameters)
# trained_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, trained_kernel)
# kta_trained = qml.kernels.target_alignment(X, Y, trained_kernel, assume_normalized_kernel=True)
# trained_svm = SVC(kernel=trained_kernel_matrix).fit(X, Y)
# accuracy_trained = get_accuracy(trained_svm, X, Y)
# trained_plot_data = cake.plot_decision_boundaries(trained_svm, plt.gca(), X, Y)
# print("-- TRAINED --")
# print("accuracy:  ", accuracy_trained)
# print("kta-value:  ", kta_trained)
#
# random_parameters = random_torch_params().numpy()
# random_kernel = lambda x1, x2: pa.kernel(x1, x2, random_parameters)
# random_kernel_matrix = lambda x1, x2: qml.kernels.kernel_matrix(x1, x2, random_kernel)
# kta_random = qml.kernels.target_alignment(X, Y, random_kernel, assume_normalized_kernel=True)
# random_svm = SVC(kernel=random_kernel_matrix).fit(X, Y)
# accuracy_random = get_accuracy(random_svm, X, Y)
# # random_plot_data = pa.plot_decision_boundaries(random_svm, plt.gca())
# print("-- UNTRAINED --")
# print("accuracy:  ", accuracy_random)
# print("kta-value:  ", kta_random)

# plotting = input("Do you wish to plot the decision boundaries? (longer waiting time) y/n")
#
# if plotting.lower() == "y":
#     plotting = 1
# elif plotting.lower() == "n":
#     plotting = 0
# else:
#     print("Invalid response. Please enter 'y' or 'n'.")