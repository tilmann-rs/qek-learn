import inspect
import q_library as qt

def count_and_record_layer_calls(func):
    # Use the inspect module to get the source code of the function
    source_code = inspect.getsource(func)

    # Define the specific layer function names you want to track
    layer_functions = ['h_layer', 'rz_layer', 'ry_layer', 'crz_ring']

    # Split the source code into lines
    lines = source_code.split('\n')

    # Initialize dictionaries to store the count and positions of layer function calls
    layer_count = {layer: 0 for layer in layer_functions}
    layer_positions = {layer: [] for layer in layer_functions}

    # Iterate through the lines and look for specific layer function calls
    for i, line in enumerate(lines, start=1):
        for layer_function in layer_functions:
            if f'{layer_function}(' in line:
                # Increment the count for the layer function
                layer_count[layer_function] += 1

                # Record the position of the layer function call
                layer_positions[layer_function].append(i)

    return layer_count, layer_positions


def generate_adjusted_block(func):
    # Use the inspect module to get the source code of the function
    source_code = inspect.getsource(func)

    # Define the specific layer function names and their corresponding adjustments
    layer_functions = {
        'h_layer': 'adj_h_layer',
        'rz_layer': 'adj_rz_layer',
        'ry_layer': 'adj_ry_layer',
        'crz_ring': 'adj_crz_ring',
    }

    # Split the source code into lines
    lines = source_code.split('\n')

    # Initialize a list to store the adjusted code
    adjusted_code = []

    # Iterate through the lines and look for specific layer function calls
    for line in lines:
        for layer_function, adjustment_function in layer_functions.items():
            if f'{layer_function}(' in line:
                # Replace the original layer function with the corresponding adjustment function
                line = line.replace(layer_function, adjustment_function)
        adjusted_code.append(line)

    # Combine the adjusted code lines into a single string
    adjusted_code_str = '\n'.join(adjusted_code)

    # Define a new function with the adjusted code
    namespace = {}
    exec(adjusted_code_str, namespace)
    adjusted_block = namespace[func.__name__]

    return adjusted_block


# Define your function
def block(num_q, x, block_params, start, state):
    state = qt.h_layer(num_q, state)
    state = qt.rz_layer(num_q, x, state, start)
    state = qt.ry_layer(num_q, block_params[0], state)
    state = qt.crz_ring(num_q, block_params[1], state)
    state = qt.h_layer(num_q, state)  # Additional h_layer call
    return state

#
# # Get the count and positions of specific layer function calls within the 'block' function
# layer_count, layer_positions = count_and_record_layer_calls(block)
#
# # Print the count and positions of each layer function
# for layer_function, count in layer_count.items():
#     print(f'{layer_function}: Count = {count}')
#     print(f'{layer_function}: Positions = {layer_positions[layer_function]}')
#


# Generate the adjusted function
adjusted_block = generate_adjusted_block(block)

print(inspect.getsource(adjusted_block))
