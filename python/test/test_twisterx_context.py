from pytwisterx.ctx.context import TwisterxContext

ctx: TwisterxContext = TwisterxContext("mpi")

print("Hello World From Rank {}, Size {}".format(ctx.get_rank(), ctx.get_world_size()))

ctx.finalize()