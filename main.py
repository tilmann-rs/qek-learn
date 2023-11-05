import sys
import time

import q_ann
import data_cake as cake

# - TIME MANAGEMENT -
start_time = time.time()


# ---
# INITIALISATION OF VARIABLES & FILE DECLARATION, USER INTERACTION
#
# epochs:       How often optimized in training process
# print_at:     Every "print_at" times the loss is printed
#
# qubits/wires: 5 used in paper, too high number of qubits (10) -> DefaultCPUAllocator: not enough memory
# block number: (how often is circuit repeated): "issue" only impair numbers work correctly by now
#
# sectors:      hoy many sectors are there in the data set / how many datapoints in each data class out of two (1 and -1)
#
# ---

NUMBER_OF_EPOCHS = int(sys.argv[1])         # around 1000
PRINT_AT = int(sys.argv[2])                 # around 1/10 of epochs

NUMBER_OF_QUBITS = int(sys.argv[3])         # originally 5, test with more
NUMBER_OF_BLOCKS = int(sys.argv[4])         # originally 5, test with more

LEARNING_RATE = float(sys.argv[5])          # around 0.05 (probably, test!)

NUMBER_OF_SECTORS = int(sys.argv[6])        # can be anything, originally 3
cake.NUMBER_OF_SECTORS = NUMBER_OF_SECTORS

resulting_parameter_file = "results/resulting_params"+"_"+str(NUMBER_OF_EPOCHS)+"_"+str(NUMBER_OF_QUBITS)+"_"+str(NUMBER_OF_BLOCKS) + \
                           "_"+str(LEARNING_RATE)+"_"+str(NUMBER_OF_SECTORS)+".txt"


# --- TRAIN & SAVE PARAMETERS ---

hyper_parameters = "hyper parameters:   " + str(NUMBER_OF_EPOCHS) + " (epochs)   " + str(NUMBER_OF_QUBITS) + " (qubits)   " \
                   + str(NUMBER_OF_BLOCKS) + " (blocks)   " + str(LEARNING_RATE) + "(learning rate)   " + str(NUMBER_OF_SECTORS) + " (number of sectors)"".res"\
                   + "\n"

with open(resulting_parameter_file, "w", encoding="utf-8") as file:
    file.write(hyper_parameters)


# TRAINING
resulting_parameters, final_kta = q_ann.train(NUMBER_OF_EPOCHS, PRINT_AT, NUMBER_OF_QUBITS, NUMBER_OF_BLOCKS, LEARNING_RATE)


with open(resulting_parameter_file, "a", encoding="utf-8") as file:
    file.write(str(resulting_parameters))
    file.write("\n")
    file.write("final kta value: "+str(final_kta))


# - TIME MANAGEMENT -
end_time = time.time()
exec_time = end_time - start_time
print(f"Execution Time: {exec_time}")

with open(resulting_parameter_file, "a", encoding="utf-8") as file:
    file.write("\n")
    file.write("Execution Time(depends highly if executed on cluster or not): "+str(exec_time)+" seconds")
