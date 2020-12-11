from pycylon import CylonContext
from pycylon.net.mpi_config import MPIConfig
from pycylon.frame import DataFrame


class distributed(object):

    def __init__(self, num_workers=1):
        self.ctx = None
        self._num_workers = num_workers
        if self._num_workers > 1:
            mpi_config = MPIConfig()
            self.ctx = CylonContext(config=mpi_config, distributed=True)
        else:
            self.ctx = CylonContext(config=None, distributed=False)

    def __enter__(self):
        print("Execution Setting")
        if self._num_workers > 1:
            print("Distributed Execution")
        else:
            print("Sequential Execution")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Finalizing Execution")
        self.ctx.finalize()

