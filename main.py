import sys
import cProfile
import time

import q_ann


# - TIME MANAGEMENT & FILE DECLARATION
start_time = time.time()
prof = cProfile.Profile()

stats_file = "stats.txt"
resulting_parameter_file = "resulting_params.txt"

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


# --- TRAIN & SAVE PARAMETERS ---
train = q_ann.train

# add cProfile.run after
resulting_parameter = train(NUMBER_OF_EPOCHS, PRINT_AT, NUMBER_OF_QUBITS, NUMBER_OF_BLOCKS, LEARNING_RATE)

# Save Data in separate file
with open(resulting_parameter_file, "w", encoding="utf-8") as file:
    file.write(str(resulting_parameter))


# - TIME MANAGEMENT -
prof.dump_stats(stats_file)
end_time = time.time()
exec_time = end_time - start_time
print(f"Execution Time: {exec_time}")
