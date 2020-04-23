from pytwisterx.common.join.config import JoinAlgorithm
from pytwisterx.common.join.config import JoinType
from pytwisterx.common.join.config import JoinConfig
from pytwisterx.utils import join as joinutils

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


joinconfig = JoinConfig(join_type="left", join_algorithm="hash", left_column_index=0, right_column_index=1)

print(joinconfig.left_index, joinconfig.right_index)

print(joinconfig.join_type, joinconfig.join_algorithm)


