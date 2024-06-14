import numpy as np
import plasticity_foam_function as sim
from mpi4py import MPI
import sys
import os
import itertools    

import alex.parameterstudy as ps

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

tensor_critical_value_hom_material =0.0

tensor_param = np.array([[0.01, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])

# sim_result = sim.run_simulation(eps_mac_param=tensor_param, scal=1.0, comm=comm)



desired_simulation_result = 0.1
def simulation_wrapper(scal_param):
            return sim.run_simulation(eps_mac_param=tensor_param, scal=scal_param, comm=comm)
scal_at_failure = ps.bisection_method(simulation_wrapper, desired_simulation_result, 0.001, 1.0, tol=0.03 * desired_simulation_result, comm=comm)
a = 1


