import sys
import time

import q_ann

# - TIME MANAGEMENT
start_time = time.time()


# ---
# INITIALISATION OF VARIABLES & FILE DECLARATION
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

resulting_parameter_file = "resulting_params"+"_"+str(NUMBER_OF_EPOCHS)+"_"+str(NUMBER_OF_QUBITS)+"_"+str(NUMBER_OF_BLOCKS)+"_"+str(LEARNING_RATE)+".txt"


# --- TRAIN & SAVE PARAMETERS ---

hyper_parameters = "hyper parameters: _ " + str(NUMBER_OF_EPOCHS) + " (epochs) _ " + str(NUMBER_OF_QUBITS) + " (qubits) _ " \
                   + str(NUMBER_OF_BLOCKS) + " (blocks) _ " + str(LEARNING_RATE) + "  (learning rate)" + ".res" + "\n"

with open(resulting_parameter_file, "w", encoding="utf-8") as file:
    file.write(hyper_parameters)

resulting_parameters = q_ann.train(NUMBER_OF_EPOCHS, PRINT_AT, NUMBER_OF_QUBITS, NUMBER_OF_BLOCKS, LEARNING_RATE)

with open(resulting_parameter_file, "a", encoding="utf-8") as file:
    file.write(str(resulting_parameters))


# - TIME MANAGEMENT -
end_time = time.time()
exec_time = end_time - start_time
print(f"Execution Time: {exec_time}")
