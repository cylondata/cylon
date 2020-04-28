from pytwisterx.net.comms import dist

dist.dist_init()
print("World Rank {}, World Size {}".format(dist.rank(), dist.size()))
dist.dist_finalize()

