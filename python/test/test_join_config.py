from pytwisterx.common.join.config import JoinAlgorithm
from pytwisterx.common.join.config import JoinType

#print(JoinAlgorithm.SORT)
# print(JoinAlgorithm.HASH)
# print(JoinType.INNER)
# print(JoinType.OUTER)
# print(JoinType.LEFT)
# print(JoinType.RIGHT)

c = JoinType.LEFT
a = JoinAlgorithm.HASH

assert (c == JoinType.LEFT)
assert (a != JoinAlgorithm.SORT)
assert (a == JoinAlgorithm.HASH)