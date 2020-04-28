from enum import Enum
from pytwisterx.common.join.config import JoinConfig

"""
TODO: If abstract config needed for the python interface
"""

class PJoinAlgorithm(Enum):
    SORT: str = "sort"
    HASH: str = "hash"


class PJoinType(Enum):
    INNER: str = "inner"
    LEFT: str = "left"
    RIGHT: str = "right"
    OUTER: str = "fullouter"
