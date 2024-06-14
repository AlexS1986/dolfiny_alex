import sys
import glob
import os
import dolfinx as dlfx
from typing import Callable
from mpi4py import MPI

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import dolfinx.fem as fem

from dolfinx.fem.petsc import LinearProblem

from petsc4py import PETSc

import numpy as np

def print_time_and_dt(t: float, dt: float):
    print(' ')
    print('==================================================')
    print('Computing solution at time = {0:.4e}'.format(t))
    print('==================================================')
    print('Current time step dt = {0:.4e}'.format(dt))
    print('==================================================')
    print(' ')
    sys.stdout.flush()
    return True
    

def print_no_convergence(dt: float):
    print('-----------------------------')
    print('!!! NO CONVERGENCE => dt: ', dt)
    print('-----------------------------')
    sys.stdout.flush()
    return True
    
def print_increasing_dt(dt: float):
    print('-----------------------------')
    print('!!! Increasing dt to: ', dt)
    print('-----------------------------')
    sys.stdout.flush()
    return True
    
def print_decreasing_dt(dt: float):
    print('-----------------------------')
    print('!!! Decreasing dt to: ', dt)
    print('-----------------------------')
    sys.stdout.flush()
    return True
    
def print_timestep_overview(iters: int, converged: bool, restart_solution: bool):
    print('-----------------------------')
    print(' No. of iterations: ', iters)
    print(' Converged:         ', converged)
    print(' Restarting:        ', restart_solution)
    print('-----------------------------')
    sys.stdout.flush()
    return True

def print_runtime(runtime: float):
    print('') 
    print('-----------------------------')
    print('elapsed time:', runtime)
    print('-----------------------------')
    print('') 
    sys.stdout.flush()
    return True
        
def prepare_newton_logfile(logfile_path: str):
    for file in glob.glob(logfile_path):
        os.remove(logfile_path)
    logfile = open(logfile_path, 'w')  
    logfile.write('# time, dt, no. iterations (for convergence) \n')
    logfile.close()
    return True
    
def write_to_newton_logfile(logfile_path: str, t: float, dt: float, iters: int):
    logfile = open(logfile_path, 'a')
    logfile.write(str(t)+'  '+str(dt)+'  '+str(iters)+'\n')
    logfile.close()
    return True

def write_runtime_to_newton_logfile(logfile_path: str, runtime: float):
    logfile = open(logfile_path, 'a')
    logfile.write('# \n')
    logfile.write('# elapsed time:  '+str(runtime)+'\n')
    logfile.write('# \n')
    logfile.close()
    return True

def default_hook():
    return

def default_hook_tdt(t,dt):
    return

def default_hook_dt(dt):
    return

def default_hook_t(t):
    return

def default_hook_all(t,dt,iters):
    return
    
def solve_with_newton_adaptive_time_stepping(domain: dlfx.mesh.Mesh,
                                             w: dlfx.fem.Function, 
                                             Tend: float,
                                             dt: float,
                                             before_first_timestep_hook: Callable = default_hook,
                                             after_last_timestep_hook: Callable = default_hook,
                                             before_each_timestep_hook: Callable = default_hook_tdt, 
                                             get_residuum_and_gateaux: Callable = default_hook_dt,
                                             get_bcs: Callable = default_hook_t,
                                             after_timestep_success_hook: Callable = default_hook_all,
                                             after_timestep_restart_hook: Callable = default_hook_all,
                                             comm: MPI.Intercomm = MPI.COMM_WORLD,
                                             print = False):
    rank = comm.Get_rank()
    
    # time stepping
    max_iters = 8
    min_iters = 4
    dt_scale_down = 0.5
    dt_scale_up = 2.0
    
    t = 0
    trestart = 0
    delta_t = dlfx.fem.Constant(domain, dt)

    before_first_timestep_hook()

    while t < Tend:
        delta_t.value = dt

        before_each_timestep_hook(t,dt)
            
        [Res, dResdw] = get_residuum_and_gateaux(delta_t)
        
        bcs = get_bcs(t)
        
        # define nonlinear problem and solver
        problem = NonlinearProblem(Res, w, bcs, dResdw)
        solver = NewtonSolver(comm, problem)
        solver.report = True
        solver.max_it = max_iters
        
        # control adaptive time adjustment
        restart_solution = False
        converged = False
        iters = max_iters + 1 # iters aalways needs to be defined
        try:
            (iters, converged) = solver.solve(w)
        except RuntimeError:
            dt = dt_scale_down*dt
            restart_solution = True
            if rank == 0 and print:
                print_no_convergence(dt)
        
        if converged and iters < min_iters and t > np.finfo(float).eps:
            dt = dt_scale_up*dt
            if rank == 0 and print:
                print_increasing_dt(dt)
        if iters >= max_iters:
            dt = dt_scale_down*dt
            restart_solution = True
            if rank == 0 and print:
                print_decreasing_dt(dt)
                
        if not converged:
            restart_solution = True

        if rank == 0 and print:    
            print_timestep_overview(iters, converged, restart_solution)
            
      

        if not(restart_solution): # TODO and converged? 
            after_timestep_success_hook(t,dt,iters)
            trestart = t
            t = t+dt
        else:
            t = trestart+dt
            after_timestep_restart_hook(t,dt,iters)
    after_last_timestep_hook()
    
    
    
    
    
    
class CustomLinearProblem(fem.petsc.LinearProblem):
        def assemble_rhs(self, u=None):
            """Assemble right-hand side and lift Dirichlet bcs.

            Parameters
            ----------
            u : dlfx.fem.Function, optional
                For non-zero Dirichlet bcs u_D, use this function to assemble rhs with the value u_D - u_{bc}
                where u_{bc} is the value of the given u at the corresponding. Typically used for custom Newton methods
                with non-zero Dirichlet bcs.
            """

            # Assemble rhs
            with self._b.localForm() as b_loc:
                b_loc.set(0)
            fem.petsc.assemble_vector(self._b, self._L)

            # Apply boundary conditions to the rhs
            x0 = [] if u is None else [u.vector]
            fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.bcs], x0=x0, scale=1.0)
            self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            x0 = None if u is None else u.vector
            fem.petsc.set_bc(self._b, self.bcs, x0, scale=1.0)

        def assemble_lhs(self):
            self._A.zeroEntries()
            fem.petsc.assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
            self._A.assemble()

        def solve_system(self):
            # Solve linear system and update ghost values in the solution
            self._solver.solve(self._b, self._x)
            self.u.x.scatter_forward()
            
            
def solve_with_custom_newton_adaptive_time_stepping(domain: dlfx.mesh.Mesh,
                                             sol: dlfx.fem.Function,
                                             dsol: dlfx.fem.Function,  
                                             Tend: float,
                                             dt: float,
                                             before_first_timestep_hook: Callable = default_hook,
                                             after_last_timestep_hook: Callable = default_hook,
                                             before_each_timestep_hook: Callable = default_hook_tdt, 
                                             get_residuum_and_gateaux: Callable = default_hook_dt,
                                             get_bcs: Callable = default_hook_t,
                                             after_iteration_hook: Callable = default_hook, 
                                             after_timestep_success_hook: Callable = default_hook_all,
                                             after_timestep_restart_hook: Callable = default_hook_all,
                                             comm: MPI.Intercomm = MPI.COMM_WORLD,
                                             print_bool = False):
    rank = comm.Get_rank()
    
    # time stepping
    max_iters = 8
    min_iters = 4
    dt_scale_down = 0.5
    dt_scale_up = 2.0
    
    t = 0.00000000001
    trestart = 0
    delta_t = dlfx.fem.Constant(domain, dt)

    before_first_timestep_hook()

    while t < Tend:
        delta_t.value = dt

        before_each_timestep_hook(t,dt)
            
       
        
        
        # define nonlinear problem and solver
        # problem = NonlinearProblem(Res, sol, bcs, dResdw)
        # solver = NewtonSolver(comm, problem)
        # solver.report = True
        # solver.max_it = max_iters
        
        # # control adaptive time adjustment
        restart_solution = False
        converged = False
        iters = max_iters + 1 # iters always needs to be defined
        try:
            [Res, dResdw] = get_residuum_and_gateaux(delta_t)
            bcs = get_bcs(t)
            Jacobi = dlfx.fem.form(dResdw)
            Residual_form = dlfx.fem.form(Res)
            parameter_handover_after_last_iteration, iters, converged = solve_with_custom_newton(Jacobi,Residual_form,sol,dsol,comm,bcs,after_iteration_hook=after_iteration_hook, print_bool=print_bool)
        except RuntimeError:
            dt = dt_scale_down*dt
            restart_solution = True
            if rank == 0 and print_bool:
                print_no_convergence(dt)
        
        if converged and iters < min_iters and t > np.finfo(float).eps:
            dt = dt_scale_up*dt
            if rank == 0 and print_bool:
                print_increasing_dt(dt)
        if iters >= max_iters:
            dt = dt_scale_down*dt
            restart_solution = True
            if rank == 0 and print_bool:
                print_decreasing_dt(dt)
                
        if not converged:
            restart_solution = True

        if rank == 0 and print_bool:    
            print_timestep_overview(iters, converged, restart_solution)

       

        if not(restart_solution):
            # after_timestep_success_hook(t,dt,iters)
            after_timestep_success_hook(t,dt,iters,parameter_handover_after_last_iteration)
            trestart = t
            t = t+dt
        else:
            t = trestart+dt
            after_timestep_restart_hook(t,dt,iters)
    after_last_timestep_hook() 
    
    
def solve_with_newton(domain, sol, dsol, Nitermax, tol,
                      load_steps,  
                      before_first_timestep_hook: Callable, 
                      before_each_timestep_hook: Callable, 
                      after_last_timestep_hook: Callable, 
                      get_residuum_and_tangent: Callable, 
                      get_bcs: Callable, 
                      after_iteration_hook: Callable, 
                      after_timestep_success_hook: Callable, 
                      comm, print_bool=False):
    
        before_first_timestep_hook()            
        for i, t in enumerate(load_steps):
            
            
            before_each_timestep_hook(t,-1.0)
            
            Residual, tangent_form = get_residuum_and_tangent(-1.0)
            
            bcs = get_bcs(t)
            # tangent_problem = LinearProblem(tangent_form,-Residual,bcs,du,petsc_options={
            # "ksp_type": "preonly",
            # "pc_type": "lu",
            # "pc_factor_mat_solver_type": "mumps",
            # })
            
            Jacobi = dlfx.fem.form(tangent_form)
            Residual_form = dlfx.fem.form(Residual)
            parameter_handover_after_last_iteration, niter, converged = solve_with_custom_newton(Jacobi,Residual_form,sol,dsol,comm,bcs,after_iteration_hook=after_iteration_hook, print_bool=print_bool)
            
            if print_bool and comm.Get_rank() == 0:
                print_timestep_overview(niter,converged=converged,restart_solution=False)
                
            after_timestep_success_hook(t,-1.0,niter,parameter_handover_after_last_iteration)
            
        #     tangent_problem = CustomLinearProblem(
        #     tangent_form,
        #     -Residual,
        #     u=du,
        #     bcs=bcs,
        #     petsc_options={
        #     "ksp_type": "preonly",
        #     "pc_type": "lu",
        #     "pc_factor_mat_solver_type": "mumps",
        #     }
        #     )
        
        
        # # compute the residual norm at the beginning of the load step
        #     tangent_problem.assemble_rhs()
        #     nRes0 = tangent_problem._b.norm()
        #     nRes = nRes0
            

        #     niter = 0
        #     while nRes / nRes0 > tol and niter < Nitermax:
        #     # update residual and tangent
        #         Residual, tangent_form = get_residuum_and_tangent()
        #         # Residual, tangent_form = alex.plasticity.get_residual_and_tangent(n, loading, as_3D_tensor(sig_np1), u_, v, eps, ds(3), dx, lmbda,mu,as_3D_tensor(N_np1),beta,H)
        #     # tangent_form = ufl.inner(eps(v), sigma_tang(eps(u_))) * dx
            
        #     # solve for the displacement correction
        #         tangent_problem.assemble_lhs()
        #         tangent_problem.solve_system()
                
        #         du = tangent_problem.solve()

        #     # update the displacement increment with the current correction
        #         parameter_handover_after_last_iteration = after_iteration_hook()

        #     # compute the new residual
        #         tangent_problem.assemble_rhs()
        #         nRes = tangent_problem._b.norm()

        #         niter += 1
                
        
        #     if niter < Nitermax:
        #         after_timestep_success_hook(t,i,niter,parameter_handover_after_last_iteration)
        #     else:
        #         raise Exception("No convergence")
        
        after_last_timestep_hook()
        
        

# def solve_with_custom_newton(jacobian, residual, sol, dsol, comm, bcs, after_iteration_hook, print_bool=False):
#     # def assemble_rhs_and_apply_bcs(jacobian, residual, uh, bcs, L):
#     #     dlfx.fem.petsc.assemble_vector(L, residual)
#     #     L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#     #     L.scale(-1)

#     #         # Compute b - J(u_D-u_(i-1))
#     #     dlfx.fem.petsc.apply_lifting(L, [jacobian], [bcs], x0=[uh.vector], scale=1)
#     #         # Set du|_bc = u_{i-1}-u_D
#     #     dlfx.fem.petsc.set_bc(L, bcs, uh.vector, 1.0)
#     #     L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
    
    
    
#     A = dlfx.fem.petsc.create_matrix(jacobian)
#     L = dlfx.fem.petsc.create_vector(residual)
    
#     solver = PETSc.KSP().create(comm)
#     solver.setOperators(A)
    
    
#     i = 0
#     max_iterations = 25
#     du_norm = []
    
#     # assemble_rhs_and_apply_bcs(jacobian, residual, uh, bcs, L)
#     # nRes0 = 100000000
    
#     converged = False
#     while i < max_iterations:

        
#     # Assemble Jacobian and residual
#         with L.localForm() as loc_L:
#             loc_L.set(0)
#         A.zeroEntries()
#         dlfx.fem.petsc.assemble_matrix(A, jacobian, bcs=bcs)
#         A.assemble()
        
#         dlfx.fem.petsc.assemble_vector(L, residual)
#         L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#         L.scale(-1)
        
#         # check for convergence
#         if i != 0:
#             if (np.isclose(np.sum(dsol.x.array),0.0) and np.isclose(np.sum(L.array),0.0) ) or correction_norm/u0 < 1e-10:
#                 converged = True
#                 break

#             # Compute b - J(u_D-u_(i-1))
#         dlfx.fem.petsc.apply_lifting(L, [jacobian], [bcs], x0=[sol.vector], scale=1)
#             # Set du|_bc = u_{i-1}-u_D
#         dlfx.fem.petsc.set_bc(L, bcs, sol.vector, 1.0)
#         L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
#         # assemble_rhs_and_apply_bcs(jacobian, residual, uh, bcs, L)
        
#         # Solve linear problem
#         solver.solve(L, dsol.vector)
#         dsol.x.scatter_forward()

#         # Update u_{i+1} = u_i + delta u_i
#         sol.x.array[:] += dsol.x.array
#         sol.x.scatter_forward()
        
#         if i == 0:
#             u0 = sol.vector.norm(0)
#         i += 1
        
#         parameters = after_iteration_hook()

#         # Compute norm of update
#         correction_norm = dsol.vector.norm(0)
        

#         # Compute L2 error comparing to the analytical solution
#         du_norm.append(correction_norm)

#         if print_bool and comm.Get_rank() == 0:
#             print(f"Converged: {converged}")
#             # print(f"Iteration {i}: Correction norm {correction_norm}")
#             print(f"Iteration {i}: Relative correction norm {correction_norm/u0 }")
#             sys.stdout.flush()
#         # if correction_norm < 1e-10:
#         #     converged = True
#         #     break
#         #     # return parameters, i
#     if converged:
#         if print_bool and comm.Get_rank() == 0:
#             print(f"Converged: { converged }")
#             # print(f"Iteration {i}: Correction norm {correction_norm}")
#             print(f"Iteration {i}: Relative correction norm {correction_norm/u0 }")
#             sys.stdout.flush()
#         return parameters, i, converged
#     # if comm.Get_rank() == 0:
#     #     raise Exception("Newton not converged")
#     return parameters, i, converged


def solve_with_custom_newton(jacobian, residual, sol, dsol, comm, bcs, after_iteration_hook, print_bool=False, max_iterations = 8):
    A = dlfx.fem.petsc.create_matrix(jacobian)
    L = dlfx.fem.petsc.create_vector(residual)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    
    i = 0
    # max_iterations = 8
    du_norm = []
    
    converged = False
    epsilon = 1e-16  # Small value to prevent division by zero

    while i < max_iterations:
        with L.localForm() as loc_L:
            loc_L.set(0)
        A.zeroEntries()
        dlfx.fem.petsc.assemble_matrix(A, jacobian, bcs=bcs)
        A.assemble()
        
        dlfx.fem.petsc.assemble_vector(L, residual)
        L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        L.scale(-1)

        # Apply lifting and set boundary conditions
        dlfx.fem.petsc.apply_lifting(L, [jacobian], [bcs], x0=[sol.vector], scale=1)
        dlfx.fem.petsc.set_bc(L, bcs, sol.vector, 1.0)
        L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        # Solve linear problem
        solver.solve(L, dsol.vector)
        dsol.x.scatter_forward()

        # Update solution
        sol.x.array[:] += dsol.x.array
        sol.x.scatter_forward()

        # Compute norms for convergence check
        correction_norm = dsol.vector.norm()
        residual_norm = np.linalg.norm(np.array(L.array))
        
        if i == 0:
            initial_residual_norm = max(residual_norm, epsilon)
            initial_solution_norm = max(sol.vector.norm(), epsilon)

        # Convergence criteria
        relative_correction_norm = correction_norm / initial_solution_norm
        relative_residual_norm = residual_norm / initial_residual_norm

        if print_bool and comm.Get_rank() == 0:
            print(f"Iteration {i}: Correction norm = {correction_norm}, Residual norm = {residual_norm}",flush=True)
            print(f"Iteration {i}: Relative correction norm = {relative_correction_norm}, Relative residual norm = {relative_residual_norm}",flush=True)
            sys.stdout.flush()


        
        i += 1
        parameters = after_iteration_hook()
        
        # Check convergence
        if relative_residual_norm < 1e-10 or relative_correction_norm < 1e-10:
            converged = True
            break

    if converged:
        if print_bool and comm.Get_rank() == 0:
            print(f"Converged after {i} iterations.")
        return parameters, i, converged

    if comm.Get_rank() == 0:
        print("Newton method did not converge within the maximum number of iterations.")
    return parameters, i, converged




