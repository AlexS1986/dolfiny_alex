from mpi4py import MPI
import sys

def bisection_method(f, target, a, b, tol=1e-6, max_iter=100, comm : MPI.Intercomm = MPI.COMM_WORLD):
    for _ in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        if comm.Get_rank() == 0:
            print(f"CURRENT SIMULATION RESULT: {fc}",flush=True)
            sys.stdout.flush()
        if abs(fc - target) < tol:
            return c
        elif fc < target:
            a = c
        else:
            b = c
    return c