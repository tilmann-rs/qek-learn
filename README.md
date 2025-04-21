### Quantum Embedding Kernel Training via Kernel Target Alignment

This repository implements a trainable, scalable quantum embedding kernel structure with variational parameters.


## Context

Quantum embedding kernels (QEKs) are a quantum kernel technique that can provide insights into learning problems by leveraging quantum feature spaces and are considered suitable for noisy intermediate-scale quantum (NISQ) devices due to their potential for shallower circuit requirements. The specific QEK implemented in this project is based on the quantum embedding structure detailed on page 10 of the paper "Training Quantum Embedding Kernels on Near-Term Quantum Computers" by Thomas Hubregtsen et al. (2021) (https://arxiv.org/abs/2105.02276). 

![grafik](https://github.com/user-attachments/assets/5b9b11c8-2b63-4c5a-a00c-95616cfffa59)

(Figure of one block component of quantum embedding circuit, modeled with IBMs quantum circuit composer)

The variational parameters of this QEK are optimized for a given dataset by maximizing the kernel-target alignment, a heuristic believed to correlate with improved achievable classification accuracy.

This repository also actively generates a synthetic dataset called Double-Cake Data, where datapoints are arranged within two non-overlapping circular sectors and are classified according to the sector they belong to.

Furthermore, the repository includes the Pennylane Tester, a utility for evaluating the performance of the trained quantum embedding models based on their optimized parameters.



## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/tilmann-rs/qek-learn.git
    cd pulse-fourier
    ```

2. Install required Python packages:
    - On Windows:
      ```bash
      pip install -r requirements.txt
      ```

3. The project is executed using main.py, it trains the neural network and requires several command-line arguments:
    ```bash
    <EPOCHS>: Number of training epochs
    <PRINT_AT>: Frequency of loss printouts during training 
    <QUBITS>: Number of qubits (e.g., 5)
    <BLOCKS>: Number of quantum circuit blocks (e.g., 5) (Odd numbers work better currently)
    <LEARNING_RATE>: Learning rate for training (e.g., 0.05)
    <SECTORS>: Number of sectors in the dataset (e.g., 5)
    ```

    Example run:
    ```bash
    python main.py 100 10 5 5 0.05 5
    ```
    


### Acknowledgements

This project was developed as part of my research internship at Intelligent Systems Group at UPV in summer 2023. The core idea and research direction were provided by Jose Antonio Pascual, and I was responsible for implementing the code and conducting the experiments.
