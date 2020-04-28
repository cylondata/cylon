import mpi4py
from mpi4py import MPI
import numpy as np
from pytwisterx.net.comms.all_to_all import AllToAll

size = 1
rank = 0
all_to_all = AllToAll(0, [0,1,2,3], [0,1,2,3], 1)

buffer = np.array([rank], dtype=np.double)
header = np.array([1,2,3,4], dtype=np.int32)

for i in range(0, size):
    all_to_all.insert(buffer, 4, i % size, header, 4)

all_to_all.finish()
while True:
    if all_to_all.is_complete():
        break


all_to_all.close()

MPI.Finalize()
