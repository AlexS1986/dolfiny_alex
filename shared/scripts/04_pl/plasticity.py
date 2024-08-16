#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import dolfinx as dlfx
import ufl
from dolfinx import default_scalar_type as scalar

import matplotlib.pyplot as plt
# import mesh_iso6892_gmshapi as mg
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
    working_folder = script_path # or script_path if local
    outputfile_graph_path = alex.os.outputfile_graph_full_path(working_folder,script_name_without_extension)

    N = 16 
    domain = dolfinx.mesh.create_unit_cube(comm,N,N,N,cell_type=dolfinx.mesh.CellType.hexahedron)

    
    # Solid: material parameters
    mu = dolfinx.fem.Constant(domain, scalar(0.5))  # [1e-9 * 1e+11 N/m^2 = 100 GPa]
    la = dolfinx.fem.Constant(domain, scalar(00.00))  # [1e-9 * 1e+10 N/m^2 =  10 GPa]
    Sy = dolfinx.fem.Constant(domain, scalar(1.0))  # initial yield stress [GPa]
    # bh = dolfinx.fem.Constant(domain, scalar(1.00))  # isotropic hardening: saturation rate  [-]
    # qh = dolfinx.fem.Constant(domain, scalar(1.5))  # isotropic hardening: saturation value [GPa]
    
    
    H = dolfinx.fem.Constant(domain, scalar(0.1))  # isotropic hardening: saturation value [GPa]


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
    urestart = dolfinx.fem.Function(Uf, name="urestart")  # displacement
    urestart.x.array[:] = np.full_like(urestart.x.array,0.0,dtype=dolfinx.default_scalar_type)
    eps_p = dolfinx.fem.Function(Tf, name="P")  # plastic strain
    eps_p_restart = dolfinx.fem.Function(Tf, name="P_restart")  # plastic strain
    eps_p_restart.x.array[:] = np.full_like(eps_p_restart.x.array,0.0,dtype=dolfinx.default_scalar_type)
    h = dolfinx.fem.Function(Hf, name="h")  # isotropic hardening
    h_restart = dolfinx.fem.Function(Hf, name="h_restart")  # isotropic hardening
    h_restart.x.array[:] = np.full_like(h_restart.x.array,0.0,dtype=dolfinx.default_scalar_type)

    u0 = dolfinx.fem.Function(Uf, name="u0")  # displacement, previous converged solution (load step)
    eps_p0 = dolfinx.fem.Function(Tf, name="P0")
    eps_p0.x.array[:] = np.zeros_like(eps_p0.x.array[:])
    
    h0 = dolfinx.fem.Function(Hf, name="h0")
    h0.x.array[:] = np.zeros_like(h0.x.array[:])

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
    m, m_restart, δm = [u, eps_p, h], [urestart, eps_p_restart, h_restart], [δu, δeps_p, δh]


    def rJ2(A):
        """Square root of J2 invariant of tensor A"""
        J2 = ufl.inner(A, A)
        rJ2 = ufl.sqrt(J2)
        return ufl.conditional(rJ2 < 1.0e-12, 0.0, rJ2)


    # Configuration gradient
    I = ufl.Identity(3)  # noqa: E741
    F = I + ufl.grad(u)  # deformation gradient as function of displacement

    # Strain measures
    E =  ufl.sym(ufl.grad(u))#1 / 2 * (F.T * F - I)  # E = E(F), total Green-Lagrange strain
    E_el = E - eps_p  # E_el = E - P, elastic strain = total strain - plastic strain

    # Stress
    S = 2 * mu * E_el + la * ufl.tr(E_el) * I  # S = S(E_el), PK2, St.Venant-Kirchhoff

    # Wrap variable around expression (for diff)
    S, h = ufl.variable(S),  ufl.variable(h)

    # Yield function
    f = rJ2(ufl.dev(S)) - ufl.sqrt(2/3)*(Sy + H*h)  # von Mises criterion (J2), with hardening

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
        + ufl.inner(δeps_p, (eps_p - eps_p0) - dλ / (2.0 * mu + 2.0 / 3.0 * H) * dgdS) * dx
        + ufl.inner(δh, (h - h0) - dλ * ufl.sqrt(2/3) / (2.0 * mu + 2.0 / 3.0 * H)) * dx # equation 10 in paper
    )
    

    # Overall form (as list of forms)
    forms = dolfiny.function.extract_blocks(form, δm)


    t = 0.0
    trestart = 0.0
    Tend = 2.0
    steps = 40
    dt = Tend/steps
    
    # time stepping
    max_iters = 20
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
    problem : dolfiny.snesblockproblem.SNESBlockProblem = dolfiny.snesblockproblem.SNESBlockProblem(forms, m, prefix=name)


    # Set up load steps
    # K = 30  # number of steps per load phase
    # load, unload = np.linspace(0.0, 1.0, num=K + 1), np.linspace(1.0, 0.0, num=K + 1)
    
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = bc.get_dimensions(domain,comm)
    
    pp.prepare_graphs_output_file(outputfile_graph_path)

    n = ufl.FacetNormal(domain)
    def reaction_force_3D(sigma_func, n: ufl.FacetNormal, ds: ufl.Measure, comm: MPI.Intercomm,):
        Rx = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[0] * ds))
        Ry = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[1] * ds))
        Rz = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[2] * ds))
        return [comm.allreduce(Rx,MPI.SUM), comm.allreduce(Ry,MPI.SUM), comm.allreduce(Rz,MPI.SUM)]
    
    atol=(x_max_all-x_min_all)*0.02 
    top_surface_tags = pp.tag_part_of_boundary(domain,bc.get_right_boundary_of_box_as_function(domain, comm,atol=atol),1)
    ds_right_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags, metadata={"quadrature_degree": quad_degree})
    
    eps_mac = dlfx.fem.Constant(domain, eps_mac_param * 0.0 * scal)
      
      
    TEN = dlfx.fem.functionspace(domain, ("DP", 0, (3, 3)))
    sigO = dolfinx.fem.Function(TEN, name="sigma")
    
    
    # Create output xdmf file -- open in Paraview with Xdmf3ReaderT
    ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
    # Write mesh, meshtags
    # ofile.write_mesh_meshtags(mesh, mts)
    ofile.write_mesh(domain)
    
    # Book-keeping of results
    results: dict[str, list[float]] = {"E": [], "S": [], "P": [], "μ": []}

    # Set up load steps
    K = 25  # number of steps per load phase
    Z = 1  # number of cycles
    load, unload = np.linspace(0.0, 1.0, num=K + 1), np.linspace(1.0, 0.0, num=K + 1)
    cycle = np.concatenate((load, unload, -load, -unload))
    cycles = np.concatenate([cycle] * Z)
    
    
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
    
    def get_dimensions(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm):
        x_min = np.min(domain.geometry.x[:,0]) 
        x_max = np.max(domain.geometry.x[:,0])   
        y_min = np.min(domain.geometry.x[:,1]) 
        y_max = np.max(domain.geometry.x[:,1])   
        z_min = np.min(domain.geometry.x[:,2]) 
        z_max = np.max(domain.geometry.x[:,2])

        # find global min/max over all mpi processes
        comm.Barrier()
        x_min_all = comm.allreduce(x_min, op=MPI.MIN)
        x_max_all = comm.allreduce(x_max, op=MPI.MAX)
        y_min_all = comm.allreduce(y_min, op=MPI.MIN)
        y_max_all = comm.allreduce(y_max, op=MPI.MAX)
        z_min_all = comm.allreduce(z_min, op=MPI.MIN)
        z_max_all = comm.allreduce(z_max, op=MPI.MAX)
        comm.Barrier()
        return x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all
    
    from functools import reduce
    
    def get_corner_of_box_as_function(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm):
        x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
        def boundary(x):
            xmin = np.isclose(x[0],x_min_all)
            xmax = np.isclose(x[0],x_max_all)
            ymin = np.isclose(x[1],y_min_all)
            ymax = np.isclose(x[1],y_max_all)
            if domain.geometry.dim == 3:
                zmin = np.isclose(x[2],z_min_all)
                zmax = np.isclose(x[2],z_max_all)
                boundaries = [xmin, ymin, zmin]
            else: #2D
                boundaries = [xmin, ymin]
            return reduce(np.logical_and, boundaries)
        return boundary
    
    for step, factor in enumerate(cycles):
    # Set current load factor
        μ.value = factor

        dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")

        # Update values for given boundary displacement
        # u_.interpolate(u_bar)
        
        
        eps_mac.value = eps_mac_param * μ.value
        
        bc_right_x = define_dirichlet_bc_from_value(domain,factor,0,right,Uf,-1)
        bc_left_x = define_dirichlet_bc_from_value(domain,-factor,0,left,Uf,-1)
        
        corner = get_corner_of_box_as_function(domain, comm)
        
        # bc_right_y = define_dirichlet_bc_from_value(domain,0.0,1,right,Uf,-1)
        bc_left_y = define_dirichlet_bc_from_value(domain,0.0,1,corner,Uf,-1)
        
        # bc_right_z = define_dirichlet_bc_from_value(domain,0.0,2,right,Uf,-1)
        bc_left_z = define_dirichlet_bc_from_value(domain,0.0,2,corner,Uf,-1)
        
        bcs = [bc_right_x, bc_left_x,  bc_left_y, bc_left_z ]
        
        # bcs = bc.get_total_linear_displacement_boundary_condition_at_box_for_incremental_formulation(
        #         domain=domain, w_n=u, functionSpace=Uf, comm=comm,eps_mac=eps_mac,subspace_idx=-1,atol=0.01)

        problem.bcs = bcs

        # Set/update boundary conditions
        # problem.bcs = [
        #     dolfinx.fem.dirichletbc(u_, surface_1_dofs_Uf),  # disp left
        #     dolfinx.fem.dirichletbc(u_, surface_2_dofs_Uf),  # disp right
        # ]

        # Solve nonlinear problem
        problem.solve()

        # Assert convergence of nonlinear solver
        problem.status(verbose=True, error_on_failure=True)

        # Post-process data
        dxg = ufl.dx
        V = dolfiny.expression.assemble(dlfx.fem.Constant(domain,1.0), dxg)
        n = ufl.as_vector([1, 0, 0])
        results["E"].append(dolfiny.expression.assemble(ufl.dot(E * n, n), dxg) / V)
        results["S"].append(dolfiny.expression.assemble(ufl.dot(S * n, n), dxg) / V)
        results["P"].append(dolfiny.expression.assemble(ufl.dot(eps_p * n, n), dxg) / V)
        results["μ"].append(factor)

        # Basic consistency checks
        # assert dolfiny.expression.assemble(dλ * df, dxg) / V < 1.0e-03, "|| dλ*df || != 0.0"
        # assert dolfiny.expression.assemble(dλ * f, dxg) / V < 1.0e-06, "|| dλ*df || != 0.0"

        # Fix: 2nd order tetrahedron
        # mesh.geometry.cmap.non_affine_atol = 1.0e-8
        # mesh.geometry.cmap.non_affine_max_its = 20

        # Write output
        u.x.scatter_forward()
        ofile.write_function(u, step)

        # Interpolate and write output
        # dolfiny.interpolation.interpolate(eps_p, Po)
        # dolfiny.interpolation.interpolate(B, Bo)
        dolfiny.interpolation.interpolate(S, So)
        dolfiny.interpolation.interpolate(h, ho)
        # ofile.write_function(Po, step)
        # ofile.write_function(Bo, step)
        ofile.write_function(So, step)
        ofile.write_function(ho, step)

        # Store stress state
        dolfiny.interpolation.interpolate(S, S0)
        
        sig_tr_expr = dlfx.fem.Expression(S, TEN.element.interpolation_points())
        sigO.interpolate(sig_tr_expr)
                    
            # ofile.write_function(sigO, t)
        Rx_top, Ry_top, Rz_top = reaction_force_3D(sigO,n=n,ds=ds_right_tagged(1),comm=comm)
        if comm.Get_rank() == 0:
            print(f"time: {t}  R_x: {Rx_top}")
        if comm.Get_rank() == 0:
            pp.write_to_graphs_output_file(outputfile_graph_path,2.0*factor, Rx_top, Ry_top, Rz_top)

        # Store primal states
        for source, target in zip([u, eps_p, h], [u0, eps_p0, h0]):
            with source.vector.localForm() as locs, target.vector.localForm() as loct:
                locs.copy(loct)
                
    if comm.Get_rank() == 0:
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Rx", "Ry", "Rz"])
    
    
    # # Adaptive load stepping
    # while t <= Tend:
    #     μ.value = t
    #     dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")

       
    #     eps_mac.value = eps_mac_param * μ.value * scal
  
    #     bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain,comm,functionSpace=Uf,
    #                                                                      eps_mac=eps_mac,
    #                                                                      subspace_idx=-1,
    #                                                                      atol=0.02*(x_max_all-x_min_all))

    #     problem.bcs = bcs
    #     restart_solution = False
    #     converged = False
        
    #     iters = max_iters + 1 # if not converged
        
    #     original_stdout = sys.stdout
    #     dummy_stream = io.StringIO()
    #     try:
    #         sys.stdout = dummy_stream
    #         problem.solve()
    #         snes : PETSc.SNES = problem.snes
            
    #         iters = snes.getIterationNumber()
    #         converged = snes.is_converged
    #         problem.status(verbose=True, error_on_failure=True)
    #         sys.stdout = original_stdout
    #     except RuntimeError:
    #         sys.stdout = original_stdout
    #         dt = dt_scale_down*dt
    #         restart_solution = True
    #         if comm.Get_rank() == 0 and print_bool:
    #             sol.print_no_convergence(dt)
                
    #     if converged and iters < min_iters and t > np.finfo(float).eps:
    #         # dt = dt_scale_up*dt
    #         if comm.Get_rank() == 0 and print_bool:
    #             sol.print_increasing_dt(dt)
    #     if iters > max_iters:
    #         dt = dt_scale_down*dt
    #         restart_solution = True
    #         if comm.Get_rank() == 0 and print_bool:
    #             sol.print_decreasing_dt(dt)
        
    #     if not converged:
    #         restart_solution = True
            
    #     if comm.Get_rank() == 0 and print_bool:
    #         sol.print_timestep_overview(iters, converged, restart_solution)
            
    #     if not restart_solution:
    #         # after load step success
    #         # Store stress state
    #         dolfiny.interpolation.interpolate(S, S0)

    #         # Store primal states
    #         for source, target in zip([u, eps_p, h], [u0, eps_p0, h0]):
    #             with source.vector.localForm() as locs, target.vector.localForm() as loct:
    #                 locs.copy(loct)
                    
            
            
    #         sig_tr_expr = dlfx.fem.Expression(S, TEN.element.interpolation_points())
    #         sigO.interpolate(sig_tr_expr)
                    
    #         # ofile.write_function(sigO, t)
    #         Rx_top, Ry_top, Rz_top = reaction_force_3D(sigO,n=n,ds=ds_right_tagged(1),comm=comm)
    #         if comm.Get_rank() == 0:
    #             print(f"time: {t}  R_x: {Rx_top}")
    #         if comm.Get_rank() == 0:
    #             pp.write_to_graphs_output_file(outputfile_graph_path,eps_mac.value[0,0], Rx_top, Ry_top, Rz_top)
            
    #         urestart.x.array[:] = u.x.array[:]
    #         h_restart.x.array[:] = h.x.array[:]
    #         eps_p_restart.x.array[:] = eps_p.x.array[:]
            
    #         trestart = t
    #         t = t+dt
    #     else:
    #         t = trestart+dt
            
    #         # after load step failure
    #         u.x.array[:] = urestart.x.array[:]
    #         h.x.array[:] = h_restart.x.array[:]
    #         eps_p.x.array[:] = eps_p_restart.x.array[:]
            
    # if comm.Get_rank() == 0:
    #     pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Rx", "Ry", "Rz"])
    
    sig_vm = le.sigvM(S)
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
        