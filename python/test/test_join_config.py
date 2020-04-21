from pytwisterx.common.join.config import Algorithm
from pytwisterx.common.join.config import JoinConfig

# print(Algorithm.SORT)
# print(Algorithm.HASH)
# print(JoinConfig.INNER)
# print(JoinConfig.OUTER)
# print(JoinConfig.LEFT)
# print(JoinConfig.RIGHT)

c = JoinConfig.LEFT
a = Algorithm.HASH

assert (c == JoinConfig.LEFT)
assert (a != Algorithm.SORT)
assert (a == Algorithm.HASH)