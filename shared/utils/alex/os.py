import os
import sys
import shutil
from datetime import datetime
from mpi4py import MPI

scripts_directory = os.path.join('/home','scripts')
resources_directory = os.path.join('/home','resources')

def logfile_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + "_log.txt")

def outputfile_graph_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + "_graphs.txt")

def outputfile_xdmf_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + ".xdmf")

def outputfile_vtk_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + ".vtk")

def mpi_print(output, rank=0):
    if rank == 0:
        print(output)
        sys.stdout.flush
    return

def set_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm,rank,size

def print_mpi_status(rank, size):
    print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
    sys.stdout.flush()


def create_timestamp_string():
    now = datetime.now()
    timestamp_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp_string

def create_results_folder(source_folder):
    timestamp = create_timestamp_string()
    results_folder = os.path.join(source_folder, timestamp + "_results")
    os.makedirs(results_folder)
    return results_folder

def copy_contents_to_results_folder(source_folder, results_folder):
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder, item)
        if os.path.isfile(source_item):
            shutil.copy(source_item, results_folder)
        elif os.path.isdir(source_item) and not item.endswith("_results"):
            shutil.copytree(source_item, os.path.join(results_folder, item))
            
            
            

