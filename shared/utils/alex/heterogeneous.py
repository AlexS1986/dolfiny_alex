import dolfinx as dlfx
import numpy as np
from array import array

def set_cell_function_heterogeneous_material(domain: dlfx.mesh.Mesh, value_inclusions: float, value_matrix: float, inclusion_cells: array, matrix_cells: array) -> dlfx.fem.Function:
    C = dlfx.fem.functionspace(domain, ("DG",0))
    cell_wise_constant_field = dlfx.fem.Function(C)
    cell_wise_constant_field.x.array[inclusion_cells] = np.full_like(inclusion_cells,value_inclusions,dtype=dlfx.default_scalar_type)
    cell_wise_constant_field.x.array[matrix_cells] = np.full_like(matrix_cells,value_matrix,dtype=dlfx.default_scalar_type)
    return cell_wise_constant_field