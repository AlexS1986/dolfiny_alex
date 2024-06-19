import numpy as np
import plasticity_foam_function as sim
from mpi4py import MPI
import sys
import os
import itertools
import alex.parameterstudy as ps
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some computations.')
parser.add_argument('total_computations', type=int, help='The total number of computations')
parser.add_argument('current_computation', type=int, help='The current computation index (starting from 0)')
args = parser.parse_args()

total_computations = args.total_computations
current_computation = args.current_computation

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

tensor_critical_value_hom_material = 1.0
desired_simulation_result = 0.1

if rank == 0:
    # Remove the file if it exists
    file_path = os.path.join(script_path, 'failure_surface.csv')
    if os.path.exists(file_path):
        os.remove(file_path)

# Define arrays
n_values = 4
values = np.linspace(-0.01, 0.01, n_values)

# Initialize arrays
sxx = values
sxy = values
sxz = values
syz = values
szz = values
syy = values

# Combine the lists into one list of lists
input_lists = [sxx, sxy, sxz, syz, szz, syy]

# Generate all combinations using itertools.product
all_combinations = list(itertools.product(*input_lists))

# Determine the share of combinations for this computation
num_combinations = len(all_combinations)
chunk_size = num_combinations // total_computations
start_index = current_computation * chunk_size
end_index = (current_computation + 1) * chunk_size if current_computation < total_computations - 1 else num_combinations

# Get the specific combinations for this computation
combinations_to_process = all_combinations[start_index:end_index]

# Initialize list to store intermediate results
intermediate_results = []

for vals in combinations_to_process:
    val_sxx, val_sxy, val_sxz, val_syz, val_szz, val_syy = vals

    tensor_param = np.array([[val_sxx, val_sxy, val_sxz],
                             [val_sxy, val_syy, val_syz],
                             [val_sxz, val_syz, val_szz]])
    if np.linalg.norm(tensor_param) < 1.0e-3:
        continue

    try:
        comm = MPI.COMM_WORLD
        def simulation_wrapper(scal_param):
            return sim.run_simulation(eps_mac_param=tensor_param, scal=scal_param, comm=comm)
        
        comm.barrier()
        scal_at_failure = ps.bisection_method(simulation_wrapper, desired_simulation_result, 0.001, 1.0, tol=0.03 * desired_simulation_result, comm=comm)
        
        principal_tensor_values_at_failure = np.linalg.eigvals(tensor_param * scal_at_failure) 

        intermediate_results.append(principal_tensor_values_at_failure)

        if rank == 0:
            print("Running computation {} of {} total".format(current_computation + 1, total_computations))
            sys.stdout.flush()

        comm.barrier()
    except Exception as e:
        if rank == 0:
            print("An error occurred in computation " + str(current_computation + 1) + " message:", e)
            print("tensor value is: \n")
            print(tensor_param)
            sys.stdout.flush()

# Write intermediate results to file
if rank == 0:
    with open(os.path.join(script_path, 'failure_surface.csv'), 'a') as file:
        for result in intermediate_results:
            if np.linalg.norm(result) < 5.0 * tensor_critical_value_hom_material:
                file.write(','.join(map(str, result)) + '\n')


# import numpy as np
# import plasticity_foam_function as sim
# from mpi4py import MPI
# import sys
# import os
# import itertools

# import alex.parameterstudy as ps

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# script_path = os.path.dirname(__file__)
# script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

# tensor_critical_value_hom_material = 1.0



# desired_simulation_result = 0.1

# if rank == 0:
#     # Remove the file if it exists
#     file_path = os.path.join(script_path, 'failure_surface.csv')
#     if os.path.exists(file_path):
#         os.remove(file_path)

# # Define arrays
# n_values = 4
# values = np.linspace(-0.01, 0.01, n_values)

# # Initialize arrays
# sxx = values
# sxy = values
# sxz = values
# syz = values
# szz = values
# syy = values

# total_iterations = len(sxx) * len(sxy) * len(sxz) * len(syz) * len(szz) * len(syy)
# computation = 1

# # Combine the lists into one list of lists
# input_lists = [sxx, sxy, sxz, syz, szz, syy]

# # Generate all combinations using itertools.product
# all_combinations = list(itertools.product(*input_lists))

# # Initialize list to store intermediate results
# intermediate_results = []

# for vals in all_combinations:
#     val_sxx, val_sxy, val_sxz, val_syz, val_szz, val_syy = vals

#     tensor_param = np.array([[val_sxx, val_sxy, val_sxz],
#                              [val_sxy, val_syy, val_syz],
#                              [val_sxz, val_syz, val_szz]])
#     if np.linalg.norm(tensor_param) < 1.0e-3:  
#         continue

#     try:
#         comm = MPI.COMM_WORLD
#         def simulation_wrapper(scal_param):
#             return sim.run_simulation(eps_mac_param=tensor_param, scal=scal_param, comm=comm)
        
#         comm.barrier()
#         scal_at_failure = ps.bisection_method(simulation_wrapper, desired_simulation_result, 0.001, 1.0, tol=0.03 * desired_simulation_result, comm=comm)
        
#         principal_tensor_values_at_failure = np.linalg.eigvals(tensor_param * scal_at_failure) 

#         intermediate_results.append(principal_tensor_values_at_failure)

#         if rank == 0:
#             print("Running computation {} of {} total".format(computation, total_iterations))
#             sys.stdout.flush()

#         computation += 1
#         comm.barrier()
#     except Exception as e:
#         if rank == 0:
#             print("An error occurred in computation " + str(computation) + " message:", e)
#             print("tensor value is: \n")
#             print(tensor_param)
#             sys.stdout.flush()
#         computation += 1

# # Write intermediate results to file
# if rank == 0:
#     with open(os.path.join(script_path, 'failure_surface.csv'), 'a') as file:
#         for result in intermediate_results:
#             if np.linalg.norm(result) < 5.0 * tensor_critical_value_hom_material:
#                 file.write(','.join(map(str, result)) + '\n')