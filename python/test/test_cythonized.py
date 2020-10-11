from pycylon.net.mpi_config import MPIConfig
from pycylon.ctx.context import XCylonContext

mpi_config = MPIConfig()

ctx = XCylonContext(mpi_config, True)

print(ctx.get_rank(), ctx.get_world_size())

ctx.finalize()
