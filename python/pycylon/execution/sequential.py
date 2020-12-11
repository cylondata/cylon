from pycylon import CylonContext


class sequential(object):
    def __init__(self):
        self.ctx = CylonContext(config=None, distributed=False)

    def __enter__(self):
        print("Sequential Execution")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Finalizing Execution")
        print(exc_type, exc_val, exc_tb)
        self.ctx.finalize()