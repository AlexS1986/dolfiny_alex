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

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver


def run_simulation(scal,eps_mac_param, comm: MPI.Intercomm):
    # references:
    # https://doi.org/10.1007/978-94-011-2860-5_66
    # https://doi.org/10.1016/j.commatsci.2012.05.062
    # https://doi.org/10.24355/dbbs.084-202112170722-0

    # comm = MPI.COMM_WORLD
    
    eps_mac_param = np.array([[1.00, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])
    
    if comm.Get_rank() == 0:
        print("Process 0 started.", flush=True)
        
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)
    dolfiny.logging.disable(level=0)

    # # Geometry and physics ansatz order
    p = 1

    script_path = os.path.dirname(__file__)
    script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
    outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)
    working_folder = script_path # or script_path if local
    logfile_path = alex.os.logfile_full_path(working_folder,script_name_without_extension)
    outputfile_graph_path = alex.os.outputfile_graph_full_path(working_folder,script_name_without_extension)

    # Foam
    # with dlfx.io.XDMFFile(comm, os.path.join(script_path,'msh2xdmf.xdmf'), 'r') as mesh_inp: 
    #         domain = mesh_inp.read_mesh()
    
    # Unit Cube     
    N = 16 
    domain = dolfinx.mesh.create_unit_cube(comm,N,N,N,cell_type=dolfinx.mesh.CellType.hexahedron)

    
    # Solid: material parameters
    mu = dolfinx.fem.Constant(domain, 1.0)  # [1e-9 * 1e+11 N/m^2 = 100 GPa]
    la = dolfinx.fem.Constant(domain, 0.0)  # [1e-9 * 1e+10 N/m^2 =  10 GPa]
    Sy = dolfinx.fem.Constant(domain, 0.250)  # initial yield stress [GPa]
    H = dolfinx.fem.Constant(domain, 0.00)  # isotropic hardening: saturation rate  [-]


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
    
    HHe = basix.ufl.element("P", domain.basix_cell(), p, shape=())

    # Define function spaces
    Uf = dolfinx.fem.functionspace(domain, Ue)
    Hf = dolfinx.fem.functionspace(domain, HHe)
    Tf = dolfinx.fem.functionspace(domain, TTe)
    
    TTf = dolfinx.fem.functionspace(domain, TTe)
    HHf = dolfinx.fem.functionspace(domain, HHe)
    
    # TTf = dolfinx.fem.functionspace(domain, TTe)
    # HHf = dolfinx.fem.functionspace(domain, HHe)

    # Define functions
    u = dolfinx.fem.Function(Uf, name="u")  # displacement
    urestart = dolfinx.fem.Function(Uf, name="urestart")  # displacement
    urestart.x.array[:] = np.full_like(urestart.x.array,0.0,dtype=dolfinx.default_scalar_type)
    
    
    def eps(u):
        return ufl.sym(ufl.nabla_grad(u))
    
    TEN = dlfx.fem.functionspace(domain, ("DP", 0, (3, 3)))
    eps_p_n = dolfinx.fem.Function(TTf, name="eps_p")  # plastic strain
    eps_p_n_out = dolfinx.fem.Function(TEN, name="eps_p_out")  # plastic strain
    eps_p_n_temp = dolfinx.fem.Function(TTf, name="eps_p_temp")  # plastic strain
    eps_p_n_restart = dolfinx.fem.Function(TTf, name="eps_p_restart")  # plastic strain
    eps_p_n_restart.x.array[:] = np.full_like(eps_p_n_restart.x.array,0.0,dtype=dolfinx.default_scalar_type)
    
    SCAL = dlfx.fem.functionspace(domain, ("DP", 0, ()))
    alpha_n = dolfinx.fem.Function(HHf, name="alpha") 
    alpha_n_out = dolfinx.fem.Function(SCAL, name="alpha_out") 
    alpha_n_temp = dolfinx.fem.Function(HHf, name="alpha_tmp") 
    alpha_n_restart = dolfinx.fem.Function(HHf, name="alpha_m1_restart")  
    alpha_n_restart.x.array[:] = np.full_like(alpha_n.x.array,0.0,dtype=dolfinx.default_scalar_type)
    
    
    f_out = dolfinx.fem.Function(HHf, name="f")
    # f_zero = dolfinx.fem.Function(HHf, name="f_zero")
    # f_zero.x.array[:] = np.zeros_like(f_zero.x.array[:]) 
    
    def sig_tr(u):
        
        return la * ufl.tr(eps(u)-eps_p_n) * ufl.Identity(domain.geometry.dim) + 2.0 * mu * (eps(u)-eps_p_n)
    
    def s_tr(u):
        return ufl.dev(sig_tr(u))
    
    def f_tr(u):
        # return ufl.sqrt(3 / 2 * ufl.inner(s_tr(u), s_tr(u))) - (Sy + H * alpha_n)
        return ufl.sqrt(ufl.inner(s_tr(u), s_tr(u))) - ufl.sqrt(2/3) * (Sy + H * alpha_n)
        
        
    def f_tr_positive(u):
        return ufl.conditional(f_tr(u) > 1.0e-12, f_tr(u), 0.0)
    
    def dGammaF(u):
        return f_tr_positive(u) / (2.0 * mu + H * 2.0 / 3.0 ) 
    
    
    def alpha_np1(u):
        return alpha_n + ufl.sqrt(2.0 / 3.0 ) * dGammaF(u)
    
    
    def direction(u):
        nenner = ufl.sqrt(ufl.inner(s_tr(u), s_tr(u)))
        direction = ufl.conditional(nenner > 1.0e-12, s_tr(u) / nenner, ufl.as_matrix([[0, 0, 0 ], [0, 0, 0], [0, 0, 0]]))
        return direction
        
    
    def eps_p_np1(u):
        return eps_p_n + dGammaF(u)* direction(u)
    
    def eps_e_np1(u):
        return eps(u) - eps_p_np1(u)
    
    def sigma_np1(u):
        return la * ufl.tr(eps_e_np1(u)) * ufl.Identity(domain.geometry.dim)  + 2.0 * mu * (eps_e_np1(u))
    
    def pot_e_np1(u):
        return ( 0.5 * ufl.inner(eps(u), sigma_np1(u)))
    
    def pot_p_np1(u):
        return ( 0.5 * H * (alpha_np1(u) -alpha_n) ** 2 )
        # return (Sy + H * alpha_np1(u)) * alpha_np1(u)
        # return ( 0.5 * H * (alpha_np1(u)) ** 2 )
    
    
    def f(u):
        # return ufl.sqrt(3 / 2 * ufl.inner(s_tr(u), s_tr(u))) - (Sy + H * alpha_n)
        return ufl.sqrt(ufl.inner(s_tr(u), s_tr(u))) - ufl.sqrt(2/3) * (Sy + H * alpha_np1(u))
    
    def penalty(u):
        gamma = 0.000000 
        return (gamma / 2.0) * ufl.max_value(0,f(u)) ** 2
        
    
    def pot(u):
        return ( pot_e_np1(u) + pot_p_np1(u)) * dx
        
    δu = ufl.TestFunction(Uf)
    δδu = ufl.TrialFunction(Uf) 
    δeps = ufl.derivative(eps(u),u,δu) 
    
    # Res = ufl.inner(sigma_np1(u),δeps) * dx
     
    Res = ufl.derivative(pot(u),u,δu)
    dResdu = ufl.derivative(Res,u,δδu)
        
        
    
        

    # u0 = dolfinx.fem.Function(Uf, name="u0")  # displacement, previous converged solution (load step)
    # eps_p0 = dolfinx.fem.Function(Tf, name="P0")
    # h0 = dolfinx.fem.Function(Hf, name="h0")

    sigO = dolfinx.fem.Function(TEN, name="sigma")  # stress, previous converged solution (load step)

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
    ofile = dolfiny.io.XDMFFile(comm, f"TEST.xdmf", "w")
    # Write mesh, meshtags
    # ofile.write_mesh_meshtags(mesh, mts)
    ofile.write_mesh(domain)
    pp.prepare_graphs_output_file(outputfile_graph_path)


    t = 0.0
    trestart = 0.0
    Tend = 1.0
    steps = 100
    dt = Tend/steps
    
    # time stepping
    max_iters = 8
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

    problem : dolfiny.snesblockproblem.SNESBlockProblem = dolfiny.snesblockproblem.SNESBlockProblem([Res], [u], prefix=name)


    
    # Set up load steps
    # K = 30  # number of steps per load phase
    # load, unload = np.linspace(0.0, 1.0, num=K + 1), np.linspace(1.0, 0.0, num=K + 1)
    
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = bc.get_dimensions(domain,comm)
    

    eps_mac = dlfx.fem.Constant(domain, eps_mac_param * 0.0)
    
    n = ufl.FacetNormal(domain)
    def reaction_force_3D(sigma_func, n: ufl.FacetNormal, ds: ufl.Measure, comm: MPI.Intercomm,):
        Rx = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[0] * ds))
        Ry = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[1] * ds))
        Rz = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[2] * ds))
        return [comm.allreduce(Rx,MPI.SUM), comm.allreduce(Ry,MPI.SUM), comm.allreduce(Rz,MPI.SUM)]
    
    atol=(x_max_all-x_min_all)*0.02 
    top_surface_tags = pp.tag_part_of_boundary(domain,bc.get_right_boundary_of_box_as_function(domain, comm,atol=atol),1)
    ds_right_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags, metadata={"quadrature_degree": quad_degree})
    
    
    def right(x):
        return np.isclose(x[0],x_max_all)
    
    def left(x):
        return np.isclose(x[0],x_min_all)
    
    def define_dirichlet_bc_from_value(domain: dlfx.mesh.Mesh,
                                                         desired_value_at_boundary: float,
                                                         coordinate_idx,
                                                         where_function,
                                                         functionSpace: dlfx.fem.FunctionSpace,
                                                         subspace_idx: int) -> dlfx.fem.DirichletBC:
        fdim = domain.topology.dim-1
        facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, where_function)
        if subspace_idx < 0:
            space = functionSpace.sub(coordinate_idx)
        else:
            space = functionSpace.sub(subspace_idx).sub(coordinate_idx)
        dofs_at_boundary = dlfx.fem.locate_dofs_topological(space, fdim, facets_at_boundary)
        bc = dlfx.fem.dirichletbc(desired_value_at_boundary,dofs_at_boundary,space)
        return bc
    
    # Adaptive load stepping
    while t <= Tend:
        μ.value = t
        dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")

        
        eps_mac.value = eps_mac_param * μ.value * scal
        ampl = 0.55
        load = ampl * t
        if t >= Tend / 2.0:
            load = ampl * Tend / 2.0 - ampl * (t - Tend/2.0)
            eps_mac.value = eps_mac_param * Tend/2.0 * scal -  eps_mac_param * (t-Tend/2.0) * scal
        
        bc_right_x = define_dirichlet_bc_from_value(domain,load,0,right,Uf,-1)
        bc_left_x = define_dirichlet_bc_from_value(domain,-load,0,left,Uf,-1)
        
        bc_right_y = define_dirichlet_bc_from_value(domain,0.0,1,right,Uf,-1)
        bc_left_y = define_dirichlet_bc_from_value(domain,0.0,1,left,Uf,-1)
        
        bc_right_z = define_dirichlet_bc_from_value(domain,0.0,2,right,Uf,-1)
        bc_left_z = define_dirichlet_bc_from_value(domain,0.0,2,left,Uf,-1)
        
        bcs = [bc_right_x, bc_left_x, bc_right_y,  bc_left_y, bc_right_z, bc_left_z ]
  
        # bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain,comm,functionSpace=Uf,
        #                                                                  eps_mac=eps_mac,
        #                                                                  subspace_idx=-1,
        #                                                                  atol=0.02*(x_max_all-x_min_all))
        
        
        # problem = NonlinearProblem(Res, u, bcs, dResdu)
        # solver = NewtonSolver(comm, problem)
        # solver.report = True
        # solver.max_it = max_iters
        
        problem.bcs = bcs
        restart_solution = False
        converged = False
        
        iters = max_iters + 1 # if not converged
        
        original_stdout = sys.stdout
        dummy_stream = io.StringIO()
        try:
            # (iters, converged) = solver.solve(u)
            sys.stdout = dummy_stream
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
            # dt = dt_scale_up*dt
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
            # dGamma = f_tr_positive(u) / (2.0 * mu + H * 2.0 / 3.0)
            
            # update on history fields
            alpha_expr = dlfx.fem.Expression(alpha_np1(u), HHf.element.interpolation_points())
            
            # alpha_n.x.array[:] = alpha_np1(u).x.array[:]
            
            alpha_n_temp.interpolate(alpha_expr)
            
            
            eps_p_expr = dlfx.fem.Expression(eps_p_n + dGammaF(u) * direction(u), TTf.element.interpolation_points())
            eps_p_n_temp.interpolate(eps_p_expr)
            
            alpha_n.x.array[:] = alpha_n_temp.x.array[:]
            eps_p_n.x.array[:] = eps_p_n_temp.x.array[:]
            
            
            
           
                
            f_out_expr = dlfx.fem.Expression(ufl.sqrt(2/3) * (Sy + H * alpha_n), HHf.element.interpolation_points())
            # f_out_expr = dlfx.fem.Expression(f_tr(u), HHf.element.interpolation_points())
            f_out.interpolate(f_out_expr)
            ofile.write_function(f_out, t)
            
            # update on history fields
            # alpha_np1_field = alpha_n + ufl.sqrt(2/3) * dGamma
            # alpha_n.x.array[:] = alpha_np1_field.x.array[:]
            
            # eps_p_n.x.array[:] = eps_p_np1(u).x.array[:]
            
            
            # then update displacements 
            urestart.x.array[:] = u.x.array[:]
            

                
                
                

            
                            # Write output
            ofile.write_function(u, t)

            # Interpolate and write output
            # dolfiny.interpolation.interpolate(eps_p, Po)
            # # dolfiny.interpolation.interpolate(B, Bo)
            # dolfiny.interpolation.interpolate(S, So)
            # dolfiny.interpolation.interpolate(h, ho)
            eps_p_n_out_expr = dlfx.fem.Expression(eps_p_n, TEN.element.interpolation_points())
            eps_p_n_out.interpolate(eps_p_n_out_expr)
            ofile.write_function(eps_p_n_out, t)
            # ofile.write_function(Bo, step)
            
            
            sig_tr_expr = dlfx.fem.Expression(sig_tr(u), TEN.element.interpolation_points())
            sigO.interpolate(sig_tr_expr)
            # dolfiny.interpolation.interpolate(sig_tr(u), sigO)
            ofile.write_function(sigO, t)
            Rx_top, Ry_top, Rz_top = reaction_force_3D(sigO,n=n,ds=ds_right_tagged(1),comm=comm)
            if comm.Get_rank() == 0:
                print(f"time: {t}  R_x: {Rx_top}")
            if comm.Get_rank() == 0:
                pp.write_to_graphs_output_file(outputfile_graph_path,load, Rx_top, Ry_top, Rz_top)
                # pp.write_to_graphs_output_file(outputfile_graph_path,eps_mac.value[0,0], Rx_top, Ry_top, Rz_top)
            
            alpha_n_out_expr = dlfx.fem.Expression(alpha_n, SCAL.element.interpolation_points())
            alpha_n_out.interpolate(alpha_n_out_expr)
            ofile.write_function(alpha_n_out, t)
            
            trestart = t
            t = t+dt
        else:
            t = trestart+dt
            
            # after load step failure
            u.x.array[:] = urestart.x.array[:]
           
    if comm.Get_rank() == 0:
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Rx", "Ry", "Rz"])
 
    
    sig_vm = le.sigvM(sigma_np1(u))
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
        