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

'''
Run test:
>> mpirun -n 4 python -m pytest --with-mpi -q python/pycylon/test/test_data_split.py
'''

from pycylon import CylonEnv, read_csv
from pycylon.net import MPIConfig
import numpy
import os
import pytest
import tempfile

@pytest.mark.mpi
def test_data_split():
    mpi_config = MPIConfig()
    env: CylonEnv = CylonEnv(config=mpi_config, distributed=True)

    rows = 100

    data_file = os.path.join(tempfile.gettempdir(), "test_split.csv")

    if env.rank == 0:
        # remove if the file already exists
        try:
            os.remove(data_file)
        except OSError:
            pass
        data = numpy.random.randint(100, size=(rows+1, 4))

        with open(data_file, 'w') as f:
            numpy.savetxt(f, data, delimiter=",", fmt='%1f')

    env.barrier()

    data_full = read_csv(data_file, slice=False, env=env)
    data = read_csv(data_file, slice=True, env=env)

    np_data = data.to_numpy()
    np_data_full = data_full.to_numpy()

    seg_size = int(rows/env.world_size)

    for i in range(0, seg_size):
        assert numpy.array_equal(
            np_data[i], np_data_full[(seg_size*env.rank)+i])

    env.barrier()
    
    if env.rank == 0:
        os.remove(data_file)
