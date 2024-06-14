from typing import Callable
import ufl
import alex.linearelastic as le
import dolfinx.fem as fem
from petsc4py import PETSc
import basix
import numpy as np


pos = lambda x: ufl.max_value(x, 0)

def constitutive_update(Δε, old_sig, alpha_n,sigY,H,lam,mu):
    sig_n = old_sig
    sig_tr = sig_n + le.sigma_as_tensor_from_epsilon(Δε,lam,mu)
    sig_dev_tr = ufl.dev(sig_tr)
    sig_eq = ufl.sqrt(3.0 / 2.0 * ufl.inner(sig_dev_tr, sig_dev_tr))
    f_tr = sig_eq - ( sigY + H * alpha_n )
    dgamma = pos(f_tr) / (3 * mu + H)
    n_elas = sig_dev_tr / sig_eq * pos(f_tr) / f_tr
    beta = 3 * mu * dgamma / sig_eq
    new_sig = sig_tr - beta * sig_dev_tr
    
    f_tr = ufl.sqrt(ufl.inner(sig_dev_tr,sig_dev_tr))- ufl.sqrt(2. / 3.) * ( sigY + H * alpha_n )
    dgamma1 = pos(f_tr) / (2. * mu + 2./3. * H)
    n_elas1 = sig_dev_tr / ufl.sqrt(ufl.inner(sig_dev_tr,sig_dev_tr)) * pos(f_tr) / f_tr
    
    new_sig = sig_tr - 2.0 * mu * dgamma1 * n_elas1  # yields the same results as above!
    # new_sig = sig_tr - beta * sig_dev_tr
    
    return new_sig, n_elas, beta, dgamma


def sigma_tang(eps, lam, mu, N_elas, beta, H):
    # N_elas = as_3D_tensor(n_elas)
    return (
        le.sigma_as_tensor_from_epsilon(eps,lam,mu)
        - 3 * mu * (3 * mu / (3 * mu + H) - beta) * ufl.inner(N_elas, eps) * N_elas
        - 2 * mu * beta * ufl.dev(eps)
    )
    
def get_residual_and_tangent(n : ufl.FacetNormal, loading, sig_np1, u_ : ufl.TestFunction, v : ufl.TrialFunction, eps: Callable, ds : ufl.Measure, dx: ufl.Measure, lam, mu, N_elas, beta, H):
    Residual = ufl.inner(eps(u_), sig_np1) * dx - ufl.inner(
        -loading * n, u_) * ds
    tangent_form = ufl.inner(eps(v), sigma_tang(eps(u_), lam, mu, N_elas, beta, H)) * dx
    return Residual, tangent_form
    
    



def get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad):
    basix_celltype = getattr(basix.CellType, domain.topology.cell_types[0].name)
    quadrature_points, weights = basix.make_quadrature(basix_celltype, deg_quad)

    map_c = domain.topology.index_map(domain.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    return quadrature_points,cells

def interpolate_quadrature(domain, cells, quadrature_points, ufl_expr):
    expr_expr = fem.Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(domain, cells)
    return expr_eval.flatten()[:]
    # function.x.array[:] = expr_eval.flatten()[:]

def to_history_field_vector_mapper(dim: int):
    def to_vect_2D(X):
        return ufl.as_vector([X[0, 0], X[1, 1], X[2, 2], X[0, 1]])
    def to_vect_3D(X):
        return ufl.as_vector([X[0, 0], X[1, 1], X[2, 2], X[1, 2], X[0, 2], X[0, 1]])
    if dim == 2:
        return to_vect_2D
    elif dim == 3:
        return to_vect_3D
    
def from_history_field_to_3D_tensor_mapper(dim: int):
    def as_3D_tensor_2D(X):
        return ufl.as_tensor([[X[0], X[3], 0], [X[3], X[1], 0], [0, 0, X[2]]])
    def as_3D_tensor_3D(X):
        return ufl.as_tensor([[X[0], X[5], X[4]], [X[5], X[1], X[3]], [X[4], X[3], X[2]]])
    if dim == 2:
        return as_3D_tensor_2D
    elif dim == 3:
        return as_3D_tensor_3D
    
def get_history_field_dimension_for_symmetric_second_order_tensor(dim: int):
    if dim == 2:
        return 4
    elif dim == 3:
        return 6
    

def eps_as_3D_tensor_function(dim: int):
    def eps_2D(v):
        e = ufl.sym(ufl.grad(v))
        return ufl.as_tensor([[e[0, 0], e[0, 1], 0], [e[0, 1], e[1, 1], 0], [0, 0, 0]])
    def eps_3D(v):
        return ufl.sym(ufl.grad(v))
    if dim == 2:
        return eps_2D
    elif dim == 3:
        return eps_3D
    
def define_internal_state_variables(gdim, domain, deg_quad, quad_scheme):  
    # W0e = basix.ufl.quadrature_element(
#     domain.basix_cell(), value_shape=(), scheme="default", degree=deg_quad
# )
# We = basix.ufl.quadrature_element(
#     domain.basix_cell(), value_shape=(alex.plasticity.get_history_field_dimension_for_symmetric_second_order_tensor(gdim),), scheme="default", degree=deg_quad
# )
    W0e = ufl.FiniteElement("Quadrature", domain.ufl_cell(), degree=deg_quad, quad_scheme=quad_scheme)
    We = ufl.VectorElement("Quadrature", domain.ufl_cell(), degree=deg_quad,dim=get_history_field_dimension_for_symmetric_second_order_tensor(gdim), quad_scheme="default")
    W0 = fem.functionspace(domain, W0e)
    W = fem.functionspace(domain, We)


    sig_np1 = fem.Function(W, name="stress_at_current_timestep")
    
    N_np1 = fem.Function(W, name="normal_to_yield_surface")
    beta = fem.Function(W0, name="beta")
    dGamma = fem.Function(W0, name="plastic_increment")
    
    # history variables 
    sig_n = fem.Function(W, name="stress_at_last_timestep")
    alpha_n = fem.Function(W0, name="Cumulative_plastic_strain")
    
    return sig_np1,sig_n,N_np1,beta,alpha_n,dGamma

def define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, quad_scheme):
    dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": deg_quad, "quadrature_scheme": quad_scheme},
    )
    
    return dx
    
    


    
    
def constitutive_update_alt(Δε, e_p_n, alpha_n,sig0,H,lam,mu):
    e_np1 = ufl.dev(Δε +  e_p_n)
    s_tr_np1 = 2.0 * mu * (e_np1 - e_p_n)
    norm_s_tr = ufl.sqrt(ufl.inner(s_tr_np1,s_tr_np1))
    f_tr = norm_s_tr- ufl.sqrt(2. / 3.) * ( sig0 + H * alpha_n )
    
    N_tr = s_tr_np1 / norm_s_tr * pos(f_tr) / f_tr
    dGamma = pos(f_tr) / (3 * mu + H)
    
    s_np1 = s_tr_np1 - 2. * mu * dGamma * N_tr
    
    
    alpha_np1 = alpha_n + ufl.sqrt(2.0 / 3.0) * dGamma
    e_p_np1 = e_p_n + dGamma * N_tr
    
    sigma_np1 = le.get_K(lam,mu) * ufl.Identity(3) * ufl.tr(Δε +  e_p_n) + 2. * mu * s_np1
    
    
    return sigma_np1, N_tr, e_p_np1, alpha_np1, dGamma, s_tr_np1


    
def sigma_tang_alt(eps,N,mu,lam,H, dGamma,s_tr_np1):
    norm_s_tr = ufl.sqrt(ufl.inner(s_tr_np1,s_tr_np1))
    return (
        le.sigma_as_tensor_from_epsilon(eps,lam,mu)
        - (2. * mu) ** 2 / (2. * mu +  H * 2. / 3.) * ufl.inner(N, eps) * N
        - (2. * mu) ** 2 / (norm_s_tr) * dGamma *  (ufl.dev(eps)-ufl.inner(N, eps) * N)
        # 
    )
    
    

