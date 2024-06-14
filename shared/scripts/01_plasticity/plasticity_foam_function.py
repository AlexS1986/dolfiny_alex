#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import dolfinx as dlfx
import ufl
from dolfinx import default_scalar_type as scalar

import matplotlib.pyplot as plt
import mesh_iso6892_gmshapi as mg
import numpy as np

import dolfiny

import os
import alex.os
import alex.boundaryconditions as bc
import alex.linearelastic as le
import alex.postprocessing as pp
import sys
import io


def run_simulation(scal,eps_mac_param, comm: MPI.Intercomm):
    # references:
    # https://doi.org/10.1007/978-94-011-2860-5_66
    # https://doi.org/10.1016/j.commatsci.2012.05.062
    # https://doi.org/10.24355/dbbs.084-202112170722-0

    # comm = MPI.COMM_WORLD
    
    if comm.Get_rank() == 0:
        print("Process 0 started.", flush=True)
        
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)
    dolfiny.logging.disable(level=0)

    # # Geometry and physics ansatz order
    p = 1

    script_path = os.path.dirname(__file__)
    script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
    outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

    with dlfx.io.XDMFFile(comm, os.path.join(script_path,'msh2xdmf.xdmf'), 'r') as mesh_inp: 
            domain = mesh_inp.read_mesh()

    
    # Solid: material parameters
    mu = dolfinx.fem.Constant(domain, scalar(100.0))  # [1e-9 * 1e+11 N/m^2 = 100 GPa]
    la = dolfinx.fem.Constant(domain, scalar(10.00))  # [1e-9 * 1e+10 N/m^2 =  10 GPa]
    Sy = dolfinx.fem.Constant(domain, scalar(0.300))  # initial yield stress [GPa]
    bh = dolfinx.fem.Constant(domain, scalar(20.00))  # isotropic hardening: saturation rate  [-]
    qh = dolfinx.fem.Constant(domain, scalar(0.100))  # isotropic hardening: saturation value [GPa]


    # Solid: load parameters
    μ = dolfinx.fem.Constant(domain, scalar(1.0))  # load factor


    # Define integration measures
    quad_degree = p
    dx = ufl.Measure(
        "dx", domain=domain, metadata={"quadrature_degree": quad_degree}
    )

    # Define elements
    Ue = basix.ufl.element("P", domain.basix_cell(), p, shape=(domain.geometry.dim,))
    He = basix.ufl.quadrature_element(domain.basix_cell(), value_shape=(), degree=quad_degree)
    Te = basix.ufl.blocked_element(He, shape=(domain.geometry.dim, domain.geometry.dim), symmetry=True)

    # Define function spaces
    Uf = dolfinx.fem.functionspace(domain, Ue)
    Tf = dolfinx.fem.functionspace(domain, Te)
    Hf = dolfinx.fem.functionspace(domain, He)

    # Define functions
    u = dolfinx.fem.Function(Uf, name="u")  # displacement
    eps_p = dolfinx.fem.Function(Tf, name="P")  # plastic strain
    h = dolfinx.fem.Function(Hf, name="h")  # isotropic hardening

    u0 = dolfinx.fem.Function(Uf, name="u0")  # displacement, previous converged solution (load step)
    eps_p0 = dolfinx.fem.Function(Tf, name="P0")
    h0 = dolfinx.fem.Function(Hf, name="h0")

    S0 = dolfinx.fem.Function(Tf, name="S0")  # stress, previous converged solution (load step)

    u_ = dolfinx.fem.Function(Uf, name="u_")  # displacement, defines state at boundary

    eps_po = dolfinx.fem.Function(
        dolfinx.fem.functionspace(domain, ("DP", 0, (3, 3))), name="P"
    )  # for output
    So = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, ("DP", 0, (3, 3))), name="S")
    ho = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, ("DP", 0)), name="h")

    δu = ufl.TestFunction(Uf)
    δeps_p = ufl.TestFunction(Tf)
    δh = ufl.TestFunction(Hf)
   
    # Define state and variation of state as (ordered) list of functions
    m, δm = [u, eps_p, h], [δu, δeps_p, δh]


    def rJ2(A):
        """Square root of J2 invariant of tensor A"""
        J2 = 1 / 2 * ufl.inner(A, A)
        rJ2 = ufl.sqrt(J2)
        return ufl.conditional(rJ2 < 1.0e-12, 0.0, rJ2)


    # Configuration gradient
    I = ufl.Identity(3)  # noqa: E741
    F = I + ufl.grad(u)  # deformation gradient as function of displacement

    # Strain measures
    E = 1 / 2 * (F.T * F - I)  # E = E(F), total Green-Lagrange strain
    E_el = E - eps_p  # E_el = E - P, elastic strain = total strain - plastic strain

    # Stress
    S = 2 * mu * E_el + la * ufl.tr(E_el) * I  # S = S(E_el), PK2, St.Venant-Kirchhoff

    # Wrap variable around expression (for diff)
    S, h = ufl.variable(S),  ufl.variable(h)

    # Yield function
    f = ufl.sqrt(3) * rJ2(ufl.dev(S)) - (Sy + h)  # von Mises criterion (J2), with hardening

    # Plastic potential
    g = f

    # Derivative of plastic potential wrt stress
    dgdS = ufl.diff(g, S)

    # Total differential of yield function, used for checks only
    # df = (
    #     +ufl.inner(ufl.diff(f, S), S - S0)
    #     + ufl.inner(ufl.diff(f, h), h - h0)
    #     # + ufl.inner(ufl.diff(f, B), B - B0)
    # )

    # Unwrap expression from variable
    S,  h = S.expression(), h.expression()
    # S, B, h = S.expression(), B.expression(), h.expression()

    # Variation of Green-Lagrange strain
    δE = dolfiny.expression.derivative(E, m, δm)

    # Plastic multiplier (J2 plasticity: closed-form solution for return-map)
    dλ = ufl.max_value(f, 0)  # ppos = MacAuley bracket

    # Weak form (as one-form)
    form = (
        ufl.inner(δE, S) * dx
        + ufl.inner(δeps_p, (eps_p - eps_p0) - dλ * dgdS) * dx
        + ufl.inner(δh, (h - h0) - dλ * bh * (qh * 1.00 - h)) * dx
    )

    # Overall form (as list of forms)
    forms = dolfiny.function.extract_blocks(form, δm)

    # Options for PETSc backend
    name = "von_mises_plasticity"
    opts = PETSc.Options(name)  # type: ignore[attr-defined]

    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_atol"] = 1.0e-12
    opts["snes_rtol"] = 1.0e-09
    opts["snes_max_it"] = 25
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"  # NOTE: this monolithic formulation is not symmetric
    opts["pc_factor_mat_solver_type"] = "mumps"
    opts.setValue('-snes_monitor', None)  # Ensure this is not set
    opts.setValue('-ksp_monitor', None)   # Ensure this is not set
    opts.setValue('-log_view', None)      # Ensure this is not set

    # Create nonlinear problem: SNES
    problem = dolfiny.snesblockproblem.SNESBlockProblem(forms, m, prefix=name)


    # Set up load steps
    K = 25  # number of steps per load phase
    load, unload = np.linspace(0.0, 1.0, num=K + 1), np.linspace(1.0, 0.0, num=K + 1)
    
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = bc.get_dimensions(domain,comm)
    # Process load steps
    for step, factor in enumerate(load):
        # Set current load factor
        μ.value = factor

        dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")

        eps_mac = dlfx.fem.Constant(domain, eps_mac_param * μ.value * scal)
        
        bcs = bc.get_total_linear_displacement_boundary_condition_at_box_for_incremental_formulation(
                domain=domain, w_n=u, functionSpace=Uf, comm=comm,eps_mac=eps_mac,subspace_idx=-1,atol=0.02*(x_max_all-x_min_all))

        problem.bcs = bcs
        
        original_stdout = sys.stdout
        dummy_stream = io.StringIO()
        sys.stdout = dummy_stream
        # Solve nonlinear problem
        problem.solve()
        sys.stdout = original_stdout
        
        # Assert convergence of nonlinear solver
        problem.status(verbose=True, error_on_failure=True)


        # Store stress state
        dolfiny.interpolation.interpolate(S, S0)

        # Store primal states
        for source, target in zip([u, eps_p, h], [u0, eps_p0, h0]):
            with source.vector.localForm() as locs, target.vector.localForm() as loct:
                locs.copy(loct)

    
    sig_vm = le.sigvM(S)
    simulation_result = pp.percentage_of_volume_above(domain,sig_vm,0.9*Sy,comm,ufl.dx,quadrature_element=True)
    return simulation_result
        