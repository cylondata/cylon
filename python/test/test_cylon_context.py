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

"""
Run test
>>  mpirun -n 2 python -m pytest --with-mpi -q python/test/test_cylon_context.py
"""

import pytest

@pytest.mark.mpi
def test_context_and_configs():
    from pycylon.net.mpi_config import MPIConfig
    from pycylon import CylonContext

    mpi_config = MPIConfig()
    ctx: CylonContext = CylonContext(config=mpi_config, distributed=True)
    ctx.add_config("compute_engine", "numpy")

    print("Hello World From Rank {}, Size {}".format(ctx.get_rank(), ctx.get_world_size()))

    assert ctx.get_config("compute_engine", "arrow") == 'numpy'

    # Note: Not needed when using PyTest with MPI
    #ctx.finalize()
