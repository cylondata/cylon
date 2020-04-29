from pytwisterx.net.comms import dist
import numpy as np
from pytwisterx.net.comms.algorithm import Communication

dist.dist_init()

size = dist.size()
rank = dist.rank()

sources = [x for x in range(size)]
targets = [x for x in range(size)]

all_to_all = Communication(rank, sources, targets, 1)

buffer = np.array([rank], dtype=np.double)
header = np.array([1,2,3,4], dtype=np.int32)

all_to_all.insert(buffer, 1, 1, header, 4)
all_to_all.wait()
all_to_all.finish()

print("World Rank {}, World Size {}".format(dist.rank(), dist.size()))
dist.dist_finalize()
