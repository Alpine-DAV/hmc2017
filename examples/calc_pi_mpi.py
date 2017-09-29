###############################################################
# mpi4py example of calculating PI using numerical integration 
###############################################################
#
# ported from C++ BLT tutorial example:
#  https://llnl.github.io/blt/index.html
#
# requires mpi4py, you can install mpi4py via conda using:
#
# {path/to/anaconda/bin}conda install mpi4py
#
#
# you can run this example using:
#
# {path/to/anaconda/bin}mpiexec -n 2 calc_pi_mpi.py 
#
###############################################################

from mpi4py import MPI
import numpy

def calc_pi_mpi(num_intervals):
    comm = MPI.COMM_WORLD
    h   = 1.0 / num_intervals;
    v_sum = numpy.array(float(0.0))
    task_id = comm.rank
    num_tasks = comm.size
    i = task_id + 1
    while(i <= num_intervals):
        x = h * (i - 0.5)
        v_sum += (4.0 / (1.0 + x*x))
        i+=num_tasks
    
    pi_local = h * v_sum
    pi = numpy.array(float(0.0))
    
    comm.Reduce(pi_local,pi, op=MPI.SUM)
    
    return pi

comm = MPI.COMM_WORLD

print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))

num_intervals = 15000

pi = calc_pi_mpi(num_intervals)

if comm.rank == 0:
    print pi


