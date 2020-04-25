import mpi4py
from mpi4py import MPI
from pytwisterx.net.comms.all_to_all import AllToAll

all_to_all = AllToAll(0, [0,1,2,3], [0,1,2,3], 1)