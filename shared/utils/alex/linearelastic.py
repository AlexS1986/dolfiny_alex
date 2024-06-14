import dolfinx as dlfx
import ufl 
from typing import Callable
import alex.phasefield
import alex.tensor
from mpi4py import MPI
import alex.util as ut

from dolfinx.fem.petsc import assemble_vector

def cmat_voigt_3D(lam: dlfx.fem.Constant, mu: dlfx.fem.Constant) -> any:
    return ufl.as_matrix([[lam+2*mu, lam, lam, 0.0, 0.0, 0.0],
                      [lam, lam+2*mu,     lam, 0.0, 0.0, 0.0],
                      [lam, lam, lam+2*mu, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,      mu,  0.0, 0.0],
                      [0.0, 0.0, 0.0,      0.0,  mu, 0.0],
                      [0.0, 0.0, 0.0,      0.0,  0.0, mu],
                      ])

# compute strain in Voigt notation
def eps_voigt_3D(u: any) -> any:
    eps = ufl.as_vector([u[0].dx(0), u[1].dx(1), u[2].dx(2), u[1].dx(2)+u[2].dx(1), u[2].dx(0)+u[0].dx(2), u[0].dx(1)+u[1].dx(0)])
    return eps


# compute strain energy in Voigt noation
def psiel_voigt(u: any, eps_funct: Callable, cmat: any) -> any:
    psiel = 0.5*ufl.dot(eps_funct(u), cmat*eps_funct(u))
    return psiel

def psiel(u: dlfx.fem.Function, sigma: any):
    return 0.5*ufl.inner(sigma, ufl.sym(ufl.grad(u)))

def get_nu(lam: float, mu: float):
    return lam/(2*(lam+mu))

def get_emod(lam: float, mu: float):
    return  mu * (3.0 * lam + 2.0 * mu) / (lam + mu)

def get_K(lam: float, mu: float):
    return  lam + 2.0 / 3.0 * mu

def get_J_3D(eshelby_as_function: Callable, n: ufl.FacetNormal, ds : ufl.Measure = ufl.ds, comm: MPI.Intracomm = MPI.COMM_WORLD):
    return alex.tensor.get_surface_integral_of_tensor_3D(eshelby_as_function,n,ds,comm)

def get_J_2D(eshelby_as_function: Callable, n: ufl.FacetNormal, ds : ufl.Measure = ufl.ds, comm: MPI.Intracomm = MPI.COMM_WORLD):
    return alex.tensor.get_surface_integral_of_tensor_2D(eshelby_as_function,n,ds,comm)

def get_J_3D_volume_integral(eshelby_as_function: Callable, dx: ufl.Measure, comm: MPI.Intracomm):
    # Jxa = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(eshelby_as_function)[0] ) * dx ) ))
    # Jya = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(eshelby_as_function)[1] ) * dx ) ))
    # Jza = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(eshelby_as_function)[2] ) * dx )))
    # return Jxa, Jya, Jza
    return alex.tensor.get_volume_integral_of_div_of_tensors_3D(eshelby_as_function,dx,comm)

def get_J_2D_volume_integral(eshelby_as_function: Callable, dx: ufl.Measure, comm: MPI.Intracomm):
    return alex.tensor.get_volume_integral_of_div_of_tensors_2D(eshelby_as_function,dx,comm)


def get_J_from_nodal_forces(eshelby_as_function: Callable, W: dlfx.fem.FunctionSpace, dx: ufl.Measure, comm: MPI.Intracomm):
    return alex.tensor.get_volume_integral_of_div_of_tensors_from_nodal_forces(eshelby_as_function,W,dx, comm)

def sigma_as_tensor(u: dlfx.fem.Function, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant ):
        eps = ufl.sym(ufl.grad(u))
        val = lam * ufl.tr(eps) * ufl.Identity(ut.get_dimension_of_function(u)) + 2*mu*eps
        return val
    
def sigma_as_tensor_from_epsilon(eps_el, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
    return lam * ufl.tr(eps_el) * ufl.Identity(3) + 2 * mu * eps_el
    
# def sigma_as_tensor2D_plane_strain(u: dlfx.fem.Function, lam:float, mu:float ):
#         eps = ufl.sym(ufl.grad(u))
#         val = lam * ufl.tr(eps) * ufl.Identity(2) + 2*mu*eps
#         return val
    
    
class StaticLinearElasticProblem:
    # Constructor method
    def __init__(self):
        self.traction = 0.0
        return 
        
    def prep_newton(self, u: any, du: ufl.TestFunction, ddu: ufl.TrialFunction, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant, dx: ufl.Measure = ufl.dx):
        def residuum(u: any, du: ufl.TestFunction, ddu: ufl.TrialFunction):
            pot = psiel(u,sigma_as_tensor(u,lam,mu))*dx + self.traction
            equi = ufl.derivative(pot, u, du)
            Res = equi
            dResdw = ufl.derivative(Res, u, ddu)
            return [ Res, dResdw]        
        return residuum(u,du,ddu)
    
    def set_traction_bc(self, sigma: any, u: dlfx.fem.Function, n: ufl.FacetNormal, ds: ufl.Measure = ufl.ds):
        self.traction = ufl.inner(ufl.dot(sigma,n),u)*ds # TODO use test function here? no since derivative?
        

def sigvM(sig):
    sdev = ufl.dev(sig)
    I2 = 0.5*ufl.inner(sdev,sdev)
    vonMises = ufl.sqrt(3*I2)
    return vonMises 

      