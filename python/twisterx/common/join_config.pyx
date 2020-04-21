from twisterx.common.join_config cimport _JoinType
from twisterx.common.join_config cimport _JoinAlgorithm
from twisterx.common.join_config cimport _JoinConfig
cimport cython


cpdef enum JoinType:
    INNER = _JoinType.INNER
    LEFT = _JoinType.LEFT
    RIGHT = _JoinType.RIGHT
    OUTER = _JoinType.OUTER

cpdef enum JoinAlgorithm:
    SORT = _JoinAlgorithm.SORT
    HASH = _JoinAlgorithm.HASH



