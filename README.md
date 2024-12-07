# Quantum Embedding Kernels with Pennylane and Double-Cake Dataset

![grafik](https://github.com/user-attachments/assets/5b9b11c8-2b63-4c5a-a00c-95616cfffa59)



This repository implements a scalable quantum embedding structure inspired by the approach detailed in the paper Training Quantum Embedding Kernels on Near-Term Quantum Computers by Thomas Hubregtsen, David Wierichs, Elies Gil-Fuster, Peter-Jan H. S. Derks, Paul K. Faehrmann, and Johannes Jakob Meyer (2021). Specifically, it utilizes the quantum embedding structure described on page 10 of the paper.


## Features

    Scalability: Users can configure the number of qubits and layers in the quantum model, enabling flexibility in adapting to different hardware constraints and problem complexities.

    Loss Function: The model optimizes the Kernel-Target Alignment (KTA), a metric that evaluates how well the kernel with its variational parameters captures the dataset’s characteristics. This helps to ensure the embedding is suited for the learning task.

    Quantum Embedding: Built to enable experimentation with quantum embedding kernels, especially in the context of near-term quantum devices.

## Data and Cake Dataset

This repository also features the creation of a synthetic dataset known as Double-Cake Data, designed to test the quantum embedding kernel. The data consists of datapoints arranged in two circular sectors, with their classification based on which sector they fall into.

## Pennylane Tester

This repository also includes the Pennylane Tester, a helper for evaluating the quantum embedding models resulting parameters.


## Arguments:

    number_of_epochs: Number of training epochs.
    number_of_qubits: Number of qubits (e.g., 5 for a simple model).
    number_of_blocks: Number of quantum blocks (layers).
    learning_rate: The learning rate used for optimization.
    number_of_sectors: The number of sectors in the double-cake dataset.
    plotting: Whether to plot the decision boundaries (0 or 1).
