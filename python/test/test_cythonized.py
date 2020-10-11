from pycylon.net.mpi_config import MPIConfig
from pycylon.ctx.context import CylonContext

mpi_config = MPIConfig()

ctx = CylonContext(mpi_config, True)

print(ctx.get_rank(), ctx.get_world_size())

ctx.finalize()
