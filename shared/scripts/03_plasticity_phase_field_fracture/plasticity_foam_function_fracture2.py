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
import alex.solution as sol
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

    pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)
    
    # Solid: material parameters
    mu = dolfinx.fem.Constant(domain, scalar(100.0))  # [1e-9 * 1e+11 N/m^2 = 100 GPa]
    la = dolfinx.fem.Constant(domain, scalar(10.00))  # [1e-9 * 1e+10 N/m^2 =  10 GPa]
    Sy = dolfinx.fem.Constant(domain, scalar(0.300))  # initial yield stress [GPa]
    H = dolfinx.fem.Constant(domain, scalar(20.00))  # isotropic hardening: saturation rate  [-]
    
    
    # Phasenfeld-Parameter
    
    
    
    
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = bc.get_dimensions(domain,comm)
    
    epsilon = dlfx.fem.Constant(domain, (y_max_all-y_min_all) / 50.0)
    def get_Gc_for_given_sig_c(sig_c, mu, epsilon):
        return (256.0 * epsilon / (27.0 * mu)) * sig_c**2
    
    Gc = dlfx.fem.Constant(domain, get_Gc_for_given_sig_c(1.1*Sy.value,mu.value,epsilon.value))
    
    eta = dlfx.fem.Constant(domain, 0.001)
    Mob = dlfx.fem.Constant(domain, 100.0)
    iMob = dlfx.fem.Constant(domain, 1.0/Mob.value)
    beta = dlfx.fem.Constant(domain, 0.1)

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
    
    TTe = basix.ufl.element("P", domain.basix_cell(), p, shape=(domain.geometry.dim,domain.geometry.dim))
    Se = basix.ufl.element("P", domain.basix_cell(), p, shape=())
    HHe = basix.ufl.element("P", domain.basix_cell(), p, shape=())

    # Define function spaces
    Wf = dlfx.fem.functionspace(domain, basix.ufl.mixed_element([Ue, Se]))
    # Uf = dolfinx.fem.functionspace(domain, Ue)
    Hf = dolfinx.fem.functionspace(domain, He)
    Tf = dolfinx.fem.functionspace(domain, Te)
    
    TTf = dolfinx.fem.functionspace(domain, TTe)
    HHf = dolfinx.fem.functionspace(domain, HHe)

    # Define functions
    w =  dlfx.fem.Function(Wf)
    wm1 =  dlfx.fem.Function(Wf)
    wrestart =  dlfx.fem.Function(Wf)
    δw = ufl.TestFunction(Wf)
    
    
    # Initialisier s=1
    wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
    wrestart.x.array[:] = wm1.x.array[:]

    # Aufspalten in Verschiebung udn Bruchfeld
    u, s = ufl.split(w)
    
    um1, sm1 = ufl.split(wm1)
    δu, δs = ufl.split(δw)
    urestart, srestart = ufl.split(wrestart)
    s_dot = ( s-sm1 ) / 1.0 # starting value wrong
    
    
    
    # u = dolfinx.fem.Function(Uf, name="u")  # displacement
    # urestart = dolfinx.fem.Function(Uf, name="urestart")  # displacement
    # urestart.x.array[:] = np.full_like(urestart.x.array,0.0,dtype=dolfinx.default_scalar_type)
    
    
    def eps(u):
        return ufl.sym(ufl.nabla_grad(u))
    
    eps_p_n = dolfinx.fem.Function(TTf, name="eps_p")  # plastic strain
    eps_p_n_restart = dolfinx.fem.Function(TTf, name="eps_p_restart")  # plastic strain
    eps_p_n_restart.x.array[:] = np.full_like(eps_p_n_restart.x.array,0.0,dtype=dolfinx.default_scalar_type)
    
    
    alpha_n = dolfinx.fem.Function(HHf, name="alpha")  # displacement
    alpha_n_restart = dolfinx.fem.Function(HHf, name="alpha_m1_restart")  # displacement
    alpha_n_restart.x.array[:] = np.full_like(alpha_n.x.array,0.0,dtype=dolfinx.default_scalar_type)
    
    
    # phase field
    def pot_s_np1(s):
        return Gc * ( 1.0 / (4.0 * epsilon) * (1-s) ** 2 + epsilon * ufl.inner(ufl.nabla_grad(s), ufl.nabla_grad(s)))
    
    # def deg(s):
    #     degrad = beta * (s ** 3 - s ** 2) + 3.0*s**2 - 2.0*s**3 + eta
    #     return degrad
    
    def deg(s):
        degrad = s ** 2 + eta
        return degrad
    
    def sig_tr(u,s):
        
        return deg(s)*la * ufl.tr(eps(u)-eps_p_n) * ufl.Identity(domain.geometry.dim) + 2.0 * mu * (eps(u)-eps_p_n)
    
    def s_tr(u,s):
        return ufl.dev(sig_tr(u,s))
    
    h = dolfinx.fem.Function(Hf, name="h")  # isotropic hardening
    h_restart = dolfinx.fem.Function(Hf, name="h_restart")  # isotropic hardening
    h_restart.x.array[:] = np.full_like(h_restart.x.array,0.0,dtype=dolfinx.default_scalar_type)
    
    def f_tr(u,s):
        return ufl.sqrt(3 / 2 * ufl.inner(s_tr(u,s), s_tr(u,s))) - (Sy + H * alpha_n)
        
        
    def f_tr_positive(u,s):
        return ufl.conditional(f_tr(u,s) > -1.0e-12, f_tr(u,s), 0.0)
    
    def alpha_np1(u):
        return alpha_n + ufl.sqrt(2.0 / 3.0 ) / (2.0 * mu + 2.0 / 3.0 * H) * f_tr_positive(u,s)
    
    
    def direction(u,s):
        nenner = ufl.sqrt(3 / 2 * ufl.inner(s_tr(u,s), s_tr(u,s)))
        direction = ufl.conditional(nenner > 1.0e-8, s_tr(u,s) / nenner, ufl.as_matrix([[0, 0, 0 ], [0, 0, 0], [0, 0, 0]]))
        return direction
        
    
    def eps_p_np1(u,s):
        return eps_p_n + 1.0 / (2.0 * mu + 2.0 / 3.0 * H) * f_tr_positive(u,s) * direction(u,s)
    
    def eps_e_np1(u,s):
        return eps(u) - eps_p_np1(u,s)
    
    def sigma_np1(u,s):
        return la * ufl.tr(eps_e_np1(u,s)) * ufl.Identity(domain.geometry.dim)  + 2.0 * mu * (eps_e_np1(u,s))
    
    def pot_e_np1(u,s):
        return ( 0.5 * deg(s)* ufl.inner(eps_e_np1(u,s), sigma_np1(u,s)))
    
    def pot_p_np1(u,s):
        return ( 0.5 * deg(s) * H * (alpha_np1(u) - alpha_n) ** 2 )
    
    
        
    
    
    def pot(u,s):
        return ( pot_e_np1(u,s) + pot_p_np1(u,s) + pot_s_np1(s)) * dx
    
    
    
    
    # δu = ufl.TestFunction(Uf)
    viscous_term = iMob*s_dot*δs*ufl.dx  
    Res = ufl.derivative(pot(u,s),u,δu) + ufl.derivative(pot(u,s),s,δs) + viscous_term
        
        
    
        

    # u0 = dolfinx.fem.Function(Uf, name="u0")  # displacement, previous converged solution (load step)
    # eps_p0 = dolfinx.fem.Function(Tf, name="P0")
    # h0 = dolfinx.fem.Function(Hf, name="h0")

    sigO = dolfinx.fem.Function(Tf, name="sigma")  # stress, previous converged solution (load step)

    # u_ = dolfinx.fem.Function(Uf, name="u_")  # displacement, defines state at boundary

    # eps_po = dolfinx.fem.Function(
    #     dolfinx.fem.functionspace(domain, ("DP", 0, (3, 3))), name="P"
    # )  # for output
    # So = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, ("DP", 0, (3, 3))), name="S")
    # ho = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, ("DP", 0)), name="h")

    # # δu = ufl.TestFunction(Uf)
    # δeps_p = ufl.TestFunction(Tf)
    # δh = ufl.TestFunction(Hf)
   
    # # Define state and variation of state as (ordered) list of functions
    # m, m_restart, δm = [u, eps_p_n, h], [urestart, eps_p_n_restart, h_restart], [δu, δeps_p, δh]


    # def rJ2(A):
    #     """Square root of J2 invariant of tensor A"""
    #     J2 = 1 / 2 * ufl.inner(A, A)
    #     rJ2 = ufl.sqrt(J2)
    #     return ufl.conditional(rJ2 < 1.0e-12, 0.0, rJ2)


    # # Configuration gradient
    # I = ufl.Identity(3)  # noqa: E741
    # F = I + ufl.grad(u)  # deformation gradient as function of displacement

    # # Strain measures
    # E = 1 / 2 * (F.T * F - I)  # E = E(F), total Green-Lagrange strain
    # E_el = E - eps_p_n  # E_el = E - P, elastic strain = total strain - plastic strain

    # # Stress
    # S = 2 * mu * E_el + la * ufl.tr(E_el) * I  # S = S(E_el), PK2, St.Venant-Kirchhoff

    # # Wrap variable around expression (for diff)
    # S, h = ufl.variable(S),  ufl.variable(h)

    # # Yield function
    # f = ufl.sqrt(3) * rJ2(ufl.dev(S)) - (Sy + h)  # von Mises criterion (J2), with hardening

    # # Plastic potential
    # g = f

    # # Derivative of plastic potential wrt stress
    # dgdS = ufl.diff(g, S)

    # # Total differential of yield function, used for checks only
    # # df = (
    # #     +ufl.inner(ufl.diff(f, S), S - S0)
    # #     + ufl.inner(ufl.diff(f, h), h - h0)
    # #     # + ufl.inner(ufl.diff(f, B), B - B0)
    # # )

    # # Unwrap expression from variable
    # S,  h = S.expression(), h.expression()
    # # S, B, h = S.expression(), B.expression(), h.expression()

    # # Variation of Green-Lagrange strain
    # δE = dolfiny.expression.derivative(E, m, δm)

    # # Plastic multiplier (J2 plasticity: closed-form solution for return-map)
    # dλ = ufl.max_value(f, 0)  # ppos = MacAuley bracket

    # # Weak form (as one-form)
    # form = (
    #     ufl.inner(δE, S) * dx
    #     + ufl.inner(δeps_p, (eps_p_n - eps_p0) - dλ * dgdS) * dx
    #     + ufl.inner(δh, (h - h0) - dλ * H * (qh * 1.00 - h)) * dx
    # )

    # # Overall form (as list of forms)
    # forms = dolfiny.function.extract_blocks(form, δm)
    # Create output xdmf file -- open in Paraview with Xdmf3ReaderT
    # ofile = dolfiny.io.XDMFFile(comm, f"TEST.xdmf", "w")
    # Write mesh, meshtags
    # ofile.write_mesh_meshtags(mesh, mts)
    # ofile.write_mesh(domain)


    t = 0.0
    trestart = 0.0
    Tend = 1.0
    steps = 20
    dt = Tend/steps
    
    # time stepping
    max_iters = 15
    min_iters = 4
    dt_scale_down = 0.5
    dt_scale_up = 2.0
    print_bool = True

    # Options for PETSc backend
    name = "von_mises_plasticity"
    opts = PETSc.Options(name)  # type: ignore[attr-defined]

    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_atol"] = 1.0e-12
    opts["snes_rtol"] = 1.0e-09
    opts["snes_max_it"] = max_iters
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"  # NOTE: this monolithic formulation is not symmetric
    opts["pc_factor_mat_solver_type"] = "mumps"
    opts.setValue('-snes_monitor', None)  # Ensure this is not set
    opts.setValue('-ksp_monitor', None)   # Ensure this is not set
    opts.setValue('-log_view', None)      # Ensure this is not set

    # Create nonlinear problem: SNES
    # problem : dolfiny.snesblockproblem.SNESBlockProblem = dolfiny.snesblockproblem.SNESBlockProblem(forms, m, prefix=name)

    problem : dolfiny.snesblockproblem.SNESBlockProblem = dolfiny.snesblockproblem.SNESBlockProblem([Res], [w], prefix=name)

    # Set up load steps
    # K = 30  # number of steps per load phase
    # load, unload = np.linspace(0.0, 1.0, num=K + 1), np.linspace(1.0, 0.0, num=K + 1)
    
    
    

    eps_mac = dlfx.fem.Constant(domain, eps_mac_param * 0.0)
      
    # Adaptive load stepping
    while t <= Tend:
        μ.value = t
        dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")

        
        eps_mac.value = eps_mac_param * μ.value * scal
        # if t >= Tend / 2.0:
        #     eps_mac.value = eps_mac_param * Tend/2.0 * scal -  eps_mac_param * (t-Tend/2.0) * scal
        
  
        bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain,comm,functionSpace=Wf,
                                                                         eps_mac=eps_mac,
                                                                         subspace_idx=0,
                                                                         atol=0.02*(x_max_all-x_min_all))

        problem.bcs = bcs
        restart_solution = False
        converged = False
        
        iters = max_iters + 1 # if not converged
        
        original_stdout = sys.stdout
        dummy_stream = io.StringIO()
        try:
            sys.stdout = dummy_stream
            
            ufl.replace(viscous_term, {s_dot: (s-sm1)/dt })
            
             
            problem.solve()
            snes : PETSc.SNES = problem.snes
            
            iters = snes.getIterationNumber()
            converged = snes.is_converged
            problem.status(verbose=True, error_on_failure=True)
            sys.stdout = original_stdout
        except RuntimeError:
            sys.stdout = original_stdout
            dt = dt_scale_down*dt
            restart_solution = True
            if comm.Get_rank() == 0 and print_bool:
                sol.print_no_convergence(dt)
                
        if converged and iters < min_iters and t > np.finfo(float).eps:
            dt = dt_scale_up*dt
            if comm.Get_rank() == 0 and print_bool:
                sol.print_increasing_dt(dt)
        if iters > max_iters:
            dt = dt_scale_down*dt
            restart_solution = True
            if comm.Get_rank() == 0 and print_bool:
                sol.print_decreasing_dt(dt)
        
        if not converged:
            restart_solution = True
            
        if comm.Get_rank() == 0 and print_bool:
            sol.print_timestep_overview(iters, converged, restart_solution)
            
        if not restart_solution:
            # after load step success
            # Store stress state
            # dolfiny.interpolation.interpolate(S, S0)

            # Store primal states
            # for source, target in zip([u, eps_p_n, h], [u0, eps_p0, h0]):
            #     with source.vector.localForm() as locs, target.vector.localForm() as loct:
            #         locs.copy(loct)
            
            
            # post-process dGamma
            dGamma = f_tr_positive(u,s) / (2 * mu + 2 / 3 * H)
            
            # update on history fields
            alpha_expr = dlfx.fem.Expression(alpha_n + ufl.sqrt(2/3)*dGamma, HHf.element.interpolation_points())
            alpha_n.interpolate(alpha_expr)
            
            
            eps_p_expr = dlfx.fem.Expression(eps_p_n + dGamma * direction(u,s), TTf.element.interpolation_points())
            eps_p_n.interpolate(eps_p_expr)
            
            # update on history fields
            # alpha_np1_field = alpha_n + ufl.sqrt(2/3) * dGamma
            # alpha_n.x.array[:] = alpha_np1_field.x.array[:]
            
            # eps_p_n.x.array[:] = eps_p_np1(u).x.array[:]
            
            
            
            # then update displacements 
            wrestart.x.array[:] = w.x.array[:]
            wm1.x.array[:] = w.x.array[:]
            
            
            pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)
                            # Write output

            pp.write_tensor_fields(domain,comm,[sigO,eps_p_n], 
                                   ["sig", "eps_p"],outputfile_xdmf_path,t)

            # Interpolate and write output
            # dolfiny.interpolation.interpolate(eps_p, Po)
            # # dolfiny.interpolation.interpolate(B, Bo)
            # dolfiny.interpolation.interpolate(S, So)
            # dolfiny.interpolation.interpolate(h, ho)
            # ofile.write_function(eps_p_n, t)
            # # ofile.write_function(Bo, step)
            # dolfiny.interpolation.interpolate(sigma_np1(u,s), sigO)
            # ofile.write_function(sigO, t)
            # ofile.write_function(alpha_n, t)
            
            trestart = t
            t = t+dt
        else:
            t = trestart+dt
            
            # after load step failure
            w.x.array[:] = wrestart.x.array[:]
            
    
    sig_vm = le.sigvM(sigma_np1(u,s))
    simulation_result = pp.percentage_of_volume_above(domain,sig_vm,0.9*Sy,comm,ufl.dx,quadrature_element=True)
    return simulation_result    
                
            
        
              
        
    #     # if not converged
        
       
    
    # # Process load steps
    # for step, factor in enumerate(load):
    #     # Set current load factor
    #     μ.value = factor

    #     dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")

    #     eps_mac = dlfx.fem.Constant(domain, eps_mac_param * μ.value * scal)
        
  
    #     bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain,comm,functionSpace=Uf,
    #                                                                      eps_mac=eps_mac,
    #                                                                      subspace_idx=-1,
    #                                                                      atol=0.02*(x_max_all-x_min_all))

    #     problem.bcs = bcs
        
        
            
            
        

    #     snes : PETSc.SNES = problem.snes
        
    #     iters = snes.getIterationNumber()
    #     converged = snes.is_converged
        
        

    #     # Store stress state
    #     dolfiny.interpolation.interpolate(S, S0)

    #     # Store primal states
    #     for source, target in zip([u, eps_p, h], [u0, eps_p0, h0]):
    #         with source.vector.localForm() as locs, target.vector.localForm() as loct:
    #             locs.copy(loct)

    
    # sig_vm = le.sigvM(S)
    # simulation_result = pp.percentage_of_volume_above(domain,sig_vm,0.9*Sy,comm,ufl.dx,quadrature_element=True)
    # return simulation_result
        