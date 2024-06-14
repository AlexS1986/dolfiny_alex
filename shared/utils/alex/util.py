import dolfinx as dlfx
import numpy as np
import ufl
import basix

def get_dimension_of_function(f: dlfx.fem.Function) -> int:
    return f.ufl_shape[0]

def print_dolfinx_version():
    print(f"DOLFINx version: {dlfx.__version__} based on GIT commit: {dlfx.git_commit_hash} of https://github.com/FEniCS/dolfinx/")


def dolfinx_cell_index(original_cell_index):
    inverse_mapping = np.empty_like(original_cell_index)
    for index, value in enumerate(original_cell_index):
        inverse_mapping[value] = index
    return inverse_mapping


def get_CG_functionspace(domain, order=1,quadrature_element=False):
    # Se = basix.ufl.element("CG", domain.basix_cell(), order) # if normal field
    if quadrature_element:
        Se = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=1, value_shape=()) # if field is quadrature element field
    else:
       Se = basix.ufl.element("CG", domain.basix_cell(), order) 
    S = dlfx.fem.functionspace(domain,Se)
    # Se = ufl.FiniteElement('CG', domain.ufl_cell(), order)
    # S = dlfx.fem.FunctionSpace(domain, Se)
    return S