from pytwisterx.common.join.config import JoinAlgorithm
from pytwisterx.common.join.config import JoinType

# print(JoinAlgorithm.SORT.value)
# print(JoinAlgorithm.HASH.value)
# print(JoinType.INNER.value)
# print(JoinType.OUTER.value)
# print(JoinType.LEFT.value)
# print(JoinType.RIGHT.value)

c = JoinType.LEFT
a = JoinAlgorithm.HASH

assert (c == JoinType.LEFT)
assert (a != JoinAlgorithm.SORT)
assert (a == JoinAlgorithm.HASH)