import dolfinx as dlfx
import numpy as np
import alex.util as ut
import os
import ufl
from scipy.ndimage import convolve

def set_cell_field_from_voxel_data(voxel_data, domain : dlfx.mesh.Mesh):
    original_cell_index = np.array(domain.topology.original_cell_index)
    Q = dlfx.fem.functionspace(domain, ("DG", 0))
    cell_field = dlfx.fem.Function(Q)
    cell_field.x.array[:] = np.array(voxel_data.flatten()[original_cell_index],dtype=dlfx.default_scalar_type)
    return cell_field


def set_voxel_data_from_field(voxel_number_x, phase, dolfinx_cell_index):
    output_voxel_from_field = np.array(phase.x.array[dolfinx_cell_index],dtype=np.uint8).reshape((voxel_number_x, voxel_number_x, voxel_number_x),order="F").flatten()
    return output_voxel_from_field


def write_field_to_voxel_data_leS(domain: dlfx.mesh.Mesh, voxel_file_out_path, voxel_number_x, field):
    original_cell_index = np.array(domain.topology.original_cell_index)
    dolfinx_cell_index_value = ut.dolfinx_cell_index(original_cell_index)
    output_voxel_from_field = set_voxel_data_from_field(voxel_number_x, field, dolfinx_cell_index_value)
    
    write_voxel_file_leS(voxel_number_x,output_voxel_from_field,voxel_file_out_path)
    # with open(voxel_file_out_path, 'w') as file:
    #         for i in range(0, len(output_voxel_from_field), voxel_number_x):
    #             line = ' '.join(map(str, output_voxel_from_field[i:i+voxel_number_x]))
    #             file.write(line + '\n')
                
def voxel_data_as_cell_tags(voxel_data, domain):
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    original_cell_index = np.array(domain.topology.original_cell_index)
    cell_tags = dlfx.mesh.meshtags(domain, domain.topology.dim, np.arange(num_cells), np.array(voxel_data.flatten()[original_cell_index],dtype=np.int32))
    return cell_tags


def read_voxel_data_leS(voxel_file_in_path, voxel_number_x, voxel_number_y, voxel_number_z):
    voxel_data = read_voxel_file_leS(voxel_file_in_path)
    
    voxel_data = list(map(float, voxel_data))
    voxel_data = np.array(voxel_data,dtype=float).reshape((voxel_number_x, voxel_number_y, voxel_number_z),order="F")
    return voxel_data

def read_voxel_file_leS(voxel_file_in_path):
    with open(voxel_file_in_path, 'r') as file:
        voxel_data = file.read().split()
    return voxel_data

def output_reduced_voxel_file_leS(script_path, script_name_without_extension, voxel_file_in_path, voxel_number_x, voxel_number_y, voxel_number_z, subarray_size, start_x, start_y, start_z, dtype=np.uint8):
    # voxel_data = read_voxel_data_leS(voxel_file_in_path,voxel_number_x)
    with open(voxel_file_in_path, 'r') as file:
             voxel_data = file.read().split()
    
    voxel_data = list(map(int, voxel_data))
    voxel_data = np.array(voxel_data,dtype=dtype).reshape((voxel_number_x, voxel_number_y, voxel_number_z))

    sub_vol = voxel_data[start_x:start_x+subarray_size, start_y:start_y+subarray_size, start_z:start_z+subarray_size].flatten()

    voxel_file_out_path_sub = os.path.join(script_path,script_name_without_extension + "_sub"+ str(subarray_size)+ ".dat")
    write_voxel_file_leS(subarray_size, sub_vol, voxel_file_out_path_sub)

def write_voxel_file_leS(voxel_dimension_x : int, flattened_voxel_array, voxel_file_out_path: str):
    with open(voxel_file_out_path, 'w') as file:
                for i in range(0, len(flattened_voxel_array), voxel_dimension_x):
                    line = ' '.join(map(str, flattened_voxel_array[i:i+voxel_dimension_x]))
                    file.write(line + '\n')

# clips a function u at a given value and assigns above and below values to another function u_clipped                
def clip (u: dlfx.fem.Function, u_clipped: dlfx.fem.Function, clip_value: float, above_value: float, below_value:float):
    expr = dlfx.fem.Expression(ufl.conditional(ufl.ge(u,clip_value),above_value,below_value),u_clipped.function_space.element.interpolation_points())
    u_clipped.interpolate(expr)
    return True


# smoothing voxel data to remove details
def smooth_3d_array(arr, iterations=1, mode='constant', cval=0.0):
    arr = arr.astype(float)
     
    # Define the 3D convolution kernel
    kernel = np.ones((3, 3, 3)) / 27.0
    
    # Perform the smoothing operation for the specified number of iterations
    for _ in range(iterations):
        arr = convolve(arr, kernel, mode=mode, cval=cval)
    return arr

# apply threshold
def apply_threshold(arr, barrier, above_value=1, below_value=0):
    # Apply the barrier condition
    thresholded_arr = np.where(arr > barrier, above_value, below_value)
    return thresholded_arr