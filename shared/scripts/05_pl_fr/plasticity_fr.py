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


from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver




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
    domain = dolfinx.mesh.create_unit_cube(comm,N,N,N,cell_type=dolfinx.mesh.CellType.tetrahedron)
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = bc.get_dimensions(domain,comm)
    
    # Solid: material parameters
    mu = dolfinx.fem.Constant(domain, scalar(0.5))  # [1e-9 * 1e+11 N/m^2 = 100 GPa]
    la = dolfinx.fem.Constant(domain, scalar(00.00))  # [1e-9 * 1e+10 N/m^2 =  10 GPa]
    Sy = dolfinx.fem.Constant(domain, scalar(0.1))  # initial yield stress [GPa]
    # bh = dolfinx.fem.Constant(domain, scalar(1.00))  # isotropic hardening: saturation rate  [-]
    # qh = dolfinx.fem.Constant(domain, scalar(1.5))  # isotropic hardening: saturation value [GPa]
    H = dolfinx.fem.Constant(domain, scalar(0.0))  # isotropic hardening: saturation value [GPa]
    
    def get_Gc_for_given_sig_c(sig_c, mu, epsilon):
        return (256.0 * epsilon / (27.0 * mu)) * sig_c**2
    # phase field
    eta = dlfx.fem.Constant(domain, 0.001)
    
    epsilon = dlfx.fem.Constant(domain, (y_max_all-y_min_all)/50.0)
    Gc = dlfx.fem.Constant(domain, get_Gc_for_given_sig_c(10.0*Sy.value,mu.value, epsilon.value))

    Mob = dlfx.fem.Constant(domain, 1000.0)
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
    Se = basix.ufl.element("P", domain.basix_cell(), p, shape=())
    He = basix.ufl.quadrature_element(domain.basix_cell(), value_shape=(), degree=quad_degree)
    Te = basix.ufl.blocked_element(He, shape=(domain.geometry.dim, domain.geometry.dim), symmetry=True)

    # Define function spaces
    Uf = dolfinx.fem.functionspace(domain, Ue)
    Sf = dolfinx.fem.functionspace(domain, Se)
    Tf = dolfinx.fem.functionspace(domain, Te)
    Hf = dolfinx.fem.functionspace(domain, He)

    # Define functions
    u = dolfinx.fem.Function(Uf, name="u")  # displacement
    urestart = dolfinx.fem.Function(Uf, name="urestart")  # displacement
    urestart.x.array[:] = np.full_like(urestart.x.array,0.0,dtype=dolfinx.default_scalar_type)
    
    
    s = dolfinx.fem.Function(Sf, name="s")  # displacement
    s.x.array[:] = np.full_like(s.x.array,1.0,dtype=dolfinx.default_scalar_type)
    sm1 = dolfinx.fem.Function(Sf, name="sm1")  # displacement
    srestart = dolfinx.fem.Function(Sf, name="srestart")  # displacement
    srestart.x.array[:] = np.full_like(srestart.x.array,1.0,dtype=dolfinx.default_scalar_type)
    sm1.x.array[:] = np.full_like(sm1.x.array,1.0,dtype=dolfinx.default_scalar_type)
    
    eps_p = dolfinx.fem.Function(Tf, name="P")  # plastic strain
    # eps_p_restart = dolfinx.fem.Function(Tf, name="P_restart")  # plastic strain
    # eps_p_restart.x.array[:] = np.full_like(eps_p_restart.x.array,0.0,dtype=dolfinx.default_scalar_type)
    alpha = dolfinx.fem.Function(Hf, name="h")  # isotropic hardening
    # h_restart = dolfinx.fem.Function(Hf, name="h_restart")  # isotropic hardening
    # h_restart.x.array[:] = np.full_like(h_restart.x.array,0.0,dtype=dolfinx.default_scalar_type)

    u0 = dolfinx.fem.Function(Uf, name="u0")  # displacement, previous converged solution (load step)
    eps_p0 = dolfinx.fem.Function(Tf, name="P0")
    eps_p0.x.array[:] = np.zeros_like(eps_p0.x.array[:])
    
    h0 = dolfinx.fem.Function(Hf, name="h0")
    h0.x.array[:] = np.zeros_like(h0.x.array[:])

    # S0 = dolfinx.fem.Function(Tf, name="S0")  # stress, previous converged solution (load step)

    # u_ = dolfinx.fem.Function(Uf, name="u_")  # displacement, defines state at boundary

    eps_po = dolfinx.fem.Function(
        dolfinx.fem.functionspace(domain, ("DP", 0, (3, 3))), name="P"
    )  # for output
    So = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, ("DP", 0, (3, 3))), name="S")
    ho = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, ("DP", 0)), name="h")

    δu = ufl.TestFunction(Uf)
    δs = ufl.TestFunction(Sf)
    δeps_p = ufl.TestFunction(Tf)
    δh = ufl.TestFunction(Hf)
   
    # Define state and variation of state as (ordered) list of functions
    m,  δm = [u, s, eps_p, alpha], [δu, δs, δeps_p, δh]
    # m,  δm = [u,  eps_p, alpha], [δu,  δeps_p, δh]

    def psisurf(s: dlfx.fem.Function, Gc: dlfx.fem.Constant, epsilon: dlfx.fem.Constant) -> any:
        psisurf = Gc.value*(((1-s)**2)/(4*epsilon.value)+epsilon.value*(ufl.dot(ufl.grad(s), ufl.grad(s))))
        return psisurf
    
    def degrad(s: any, eta: dlfx.fem.Constant) -> any:
        degrad = beta * (s ** 3 - s ** 2) + 3.0*s**2 - 2.0*s**3 + eta
        return degrad
    
    def degrad_div(s: any, eta: dlfx.fem.Constant) -> any:
        degrad = beta * (3.0 * s ** 2 - 2.0 * s) + 6.0*s - 6*s**2 + eta
        return degrad


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
    S = degrad(s,eta) * ( 2.0 * mu * E_el + la * ufl.tr(E_el) * I )  # S = S(E_el), PK2, St.Venant-Kirchhoff
    S_star = 2 * mu * E_el + la * ufl.tr(E_el) * I

    # Wrap variable around expression (for diff)
    S, alpha = ufl.variable(S),  ufl.variable(alpha)
    S_star = ufl.variable(S_star)

    # Yield function
    f = degrad(s,eta) * (rJ2(ufl.dev(S_star)) - ufl.sqrt(2/3)*(Sy + H*alpha))  # von Mises criterion (J2), with hardening

    f_star = rJ2(ufl.dev(S)) - ufl.sqrt(2/3)*(Sy + H*alpha)

    # Plastic potential
    g = f_star

    # Derivative of plastic potential wrt stress
    dgdS = ufl.diff(g, S)

    # Unwrap expression from variable
    S,  alpha = S.expression(), alpha.expression()
    S_star = S_star.expression()

    # Variation of Green-Lagrange strain
    δE = dolfiny.expression.derivative(E, m, δm)

    # Plastic multiplier (J2 plasticity: closed-form solution for return-map)
    dλ = ufl.max_value(f, 0)  # ppos = MacAuley bracket
    
    pot = 0.5 * ufl.inner(S,E) + psisurf(s,Gc,epsilon) + degrad(s,eta) * ( Sy + 0.5 * H * alpha) * alpha
    sdrive = ufl.derivative(pot * dx, s, δs)
    
    dtt = dolfinx.fem.Constant(domain,0.1)
    rate_term = iMob*(s-sm1) / dtt* δs*dx
    
    # Weak form (as one-form)
    form = (
        # sdrive + rate_term +
        (δs * ( iMob*(s-sm1) / dtt ) + ufl.inner(ufl.grad(δs), 2.0*epsilon*Gc*ufl.grad(s)) + 
        δs * ( degrad_div(s,eta) * (0.5* ufl.inner(S_star,E) + ( Sy + 0.5 * H * alpha) * alpha )  - Gc / (2.0 *epsilon) * (1 - s))) * dx + 
        ufl.inner(δE, S) * dx + 
        + ufl.inner(δeps_p, (eps_p - eps_p0) - dλ / (2.0 * mu + 2.0 / 3.0 * H) * dgdS) * dx
        + ufl.inner(δh, (alpha - h0) - dλ * ufl.sqrt(2/3) / (2.0 * mu + 2.0 / 3.0 * H)) * dx # equation 10 in paper
    )
    
    # Overall form (as list of forms)
    forms = dolfiny.function.extract_blocks(form, δm)


    # δu = ufl.TestFunction(Uf)
    # δδu = ufl.TrialFunction(Uf) 
    # δeps = ufl.derivative(eps(u),u,δu) 
    
    # Res = ufl.inner(sigma_np1(u),δeps) * dx
    # δδu = ufl.TrialFunction(Uf)
    # δδeps_p = ufl.TrialFunction(Tf)
    # δδh = ufl.TrialFunction(Hf)
   
    # Define state and variation of state as (ordered) list of functions
    # δδm = [δδu, δδeps_p, δδh]
    
    
     
    # Res = ufl.inner(δE, S) * dx + ufl.inner(δeps_p, (eps_p - eps_p0) - dλ / (2.0 * mu + 2.0 / 3.0 * H) * dgdS) * dx + ufl.inner(δh, (h - h0) - dλ * ufl.sqrt(2/3) / (2.0 * mu + 2.0 / 3.0 * H)) * dx
    # dResdm = ufl.derivative(Res,m,δδm)
    

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
    
    VEC = dlfx.fem.functionspace(domain, ("DP", 0, (3,)))
    uO = dolfinx.fem.Function(VEC, name="sigma")
    
    
    ofile = dolfiny.io.XDMFFile(comm, outputfile_xdmf_path, "w")
    ofile.write_mesh(domain)
    # pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)
    
    # Set up load steps
    # K = 25  # number of steps per load phase
    # Z = 1  # number of cycles
    # load, unload = np.linspace(0.0, 1.0, num=K + 1), np.linspace(1.0, 0.0, num=K + 1)
    # cycle = np.concatenate((load, unload, -load, -unload))
    # cycles = np.concatenate([cycle] * Z)
    
    
    def right(x):
        return np.isclose(x[0],x_max_all)
    
    def left(x):
        return np.isclose(x[0],x_min_all)
    
    def u_bar(x):
        return μ.value * np.array([0.1 * x[0], 0.0 * x[1], 0.0 * x[2]])
    
    
    # Adaptive load stepping
    while t <= Tend:
        μ.value = t
        dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")
        dtt.value = dt
        
        
        eps_mac.value = eps_mac_param * μ.value * scal
        ampl = 0.55
        load = ampl * t
        if t >= Tend / 2.0:
            load = ampl * Tend / 2.0 - ampl * (t - Tend/2.0)
            eps_mac.value = eps_mac_param * Tend/2.0 * scal -  eps_mac_param * (t-Tend/2.0) * scal
        
        bc_right_x = bc.define_dirichlet_bc_from_value(domain,load,0,right,Uf,-1)
        bc_left_x = bc.define_dirichlet_bc_from_value(domain,-load,0,left,Uf,-1)
        
        # bc_right_y = bc.define_dirichlet_bc_from_value(domain,0.0,1,right,Uf,-1)
        # bc_left_y = bc.define_dirichlet_bc_from_value(domain,0.0,1,left,Uf,-1)
        
        # bc_right_z = bc.define_dirichlet_bc_from_value(domain,0.0,2,right,Uf,-1)
        # bc_left_z = bc.define_dirichlet_bc_from_value(domain,0.0,2,left,Uf,-1)
        
        
        corner = bc.get_corner_of_box_as_function(domain,comm)
        bc_corner_y= bc.define_dirichlet_bc_from_value(domain,0.0,1,corner,Uf,-1)
        bc_corner_z= bc.define_dirichlet_bc_from_value(domain,0.0,2,corner,Uf,-1)
        
        # bcs = [bc_right_x, bc_left_x, bc_right_y, bc_left_y, bc_right_z, bc_left_z ]
        
        bcs = [bc_right_x, bc_left_x, bc_corner_y, bc_corner_z ]
  
        # bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain,comm,functionSpace=Uf,
        #                                                                  eps_mac=eps_mac,
        #                                                                  subspace_idx=-1,
        #                                                                  atol=0.02*(x_max_all-x_min_all))
        
        
        # problem = NonlinearProblem(Res, m, bcs, dResdm)
        # solver = NewtonSolver(comm, problem)
        # solver.report = True
        # solver.max_it = max_iters
        
        problem.bcs = bcs
            # Update values for given boundary displacement
        # u_.interpolate(u_bar)

        # fdim = domain.topology.dim-1
        # left_facets = dlfx.mesh.locate_entities_boundary(domain, fdim, left)
        # left_dofs_Uf = dolfinx.fem.locate_dofs_topological(Uf,fdim,left_facets)
        
        # right_facets = dlfx.mesh.locate_entities_boundary(domain, fdim, right)
        # right_dofs_Uf = dolfinx.fem.locate_dofs_topological(Uf,fdim,right_facets)
        # # Set/update boundary conditions
        # problem.bcs = [
        #     dolfinx.fem.dirichletbc(u_, left_dofs_Uf),  # disp left
        #     dolfinx.fem.dirichletbc(u_, right_dofs_Uf),  # disp right
        # ]
        
        
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
            # Assert convergence of nonlinear solver
            problem.status(verbose=True, error_on_failure=True)


            # u_expr = dlfx.fem.Expression(u, uO.function_space.element.interpolation_points())
            # uO.interpolate(u_expr)
            # with dolfinx.io.XDMFFile(comm, outputfile_xdmf_path, "a") as xdmf:
            #     xdmf.write_function(uO,t)
                
            ofile.write_function(u, t)
            ofile.write_function(s, t)
            
            #pp.write_vector_field(domain,outputfile_xdmf_path,u,t,comm)
            # pp.write_vector_fields(domain,comm,[u], ["u"], outputfile_xdmf_path,t)


            # for quadrature element
            h_expr = dlfx.fem.Expression(alpha, ho.function_space.element.interpolation_points())
            ho.interpolate(h_expr)
            pp.write_field(domain,outputfile_xdmf_path,ho,t,comm)
            
            pp.write_tensor_fields(domain,comm,[S,eps_p], ["S", "eps_p"],outputfile_xdmf_path,t)

            sig_expr = dlfx.fem.Expression(S, sigO.function_space.element.interpolation_points())
            sigO.interpolate(sig_expr)    
            
            Rx_top, Ry_top, Rz_top = reaction_force_3D(sigO,n=n,ds=ds_right_tagged(1),comm=comm)
            if comm.Get_rank() == 0:
                print(f"time: {t}  R_x: {Rx_top}")
            if comm.Get_rank() == 0:
                pp.write_to_graphs_output_file(outputfile_graph_path,2.0*load, Rx_top, Ry_top, Rz_top)

            # Store primal states
            for source, target in zip([u, eps_p, alpha], [u0, eps_p0, h0]):
                with source.vector.localForm() as locs, target.vector.localForm() as loct:
                    locs.copy(loct)
            
            trestart = t
            urestart.x.array[:] = u.x.array[:]
            srestart.x.array[:] = s.x.array[:]
            sm1.x.array[:] = s.x.array[:]
            t = t+dt
        else:
            t = trestart+dt
            
            # after load step failure
            u.x.array[:] = urestart.x.array[:]
            s.x.array[:] = srestart.x.array[:]
    
    ofile.close()
             
    if comm.Get_rank() == 0:
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Rx", "Ry", "Rz"])
    

    sig_vm = le.sigvM(S)
    simulation_result = pp.percentage_of_volume_above(domain,sig_vm,0.9*Sy,comm,ufl.dx,quadrature_element=True)
    return simulation_result    
                
            
        
        