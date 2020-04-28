from pytwisterx.net.comms import dist
import numpy as np
from pytwisterx.net.comms.algorithm import Communication

dist.dist_init()



size = dist.size()
rank = dist.rank()
all_to_all = Communication(rank, [0,1,2,3], [0,1,2,3], 1)

buffer = np.array([rank], dtype=np.double)
header = np.array([1,2,3,4], dtype=np.int32)


#
all_to_all.insert(buffer, 1, 1, header, 4)
all_to_all.wait()
all_to_all.finish()
#
# all_to_all.finish()
# while True:
#     if all_to_all.is_complete():
#         break
#
#
# all_to_all.close()


print("World Rank {}, World Size {}".format(dist.rank(), dist.size()))
dist.dist_finalize()
