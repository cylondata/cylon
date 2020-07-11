##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##

from pycylon.net import dist
import numpy as np
from pycylon.net.comms import Communication

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
