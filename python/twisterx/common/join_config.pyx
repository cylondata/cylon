from twisterx.common.join_config cimport CJoinType
from twisterx.common.join_config cimport CJoinAlgorithm
from twisterx.common.join_config cimport CJoinConfig
#from pytwisterx.common.join.config import JoinType
#from pytwisterx.common.join.config import JoinAlgorithm
cimport cython
from enum import Enum


class PJoinAlgorithm(Enum):
    SORT = "sort"
    HASH = "hash"


class PJoinType(Enum):
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "fullouter"


cpdef enum JoinType:
    INNER = CJoinType.INNER
    LEFT = CJoinType.LEFT
    RIGHT = CJoinType.RIGHT
    OUTER = CJoinType.OUTER

cpdef enum JoinAlgorithm:
    SORT = CJoinAlgorithm.SORT
    HASH = CJoinAlgorithm.HASH

cdef class JoinConfig:
    cdef CJoinConfig *jcPtr
    cdef CJoinType jtPtr
    cdef CJoinAlgorithm jaPtr

    def __cinit__(self, join_type: str, join_algorithm: str, left_column_index: int, right_column_index: int):
        if join_type is not None and join_algorithm is not None and left_column_index is not None and right_column_index is not None:
            self._get_join_config(join_type=join_type, join_algorithm=join_algorithm,
                                  left_column_index=left_column_index,
                                  right_column_index=right_column_index)

    cdef _get_join_config(self, join_type: str, join_algorithm: str, left_column_index: int,
                          right_column_index: int):
        if left_column_index is None or right_column_index is None:
            raise Exception("Join Column index not provided")

        if join_algorithm is None:
            join_algorithm = PJoinAlgorithm.HASH.value

        if join_algorithm == PJoinAlgorithm.HASH.value:

            if join_type == PJoinType.INNER.value:
                self.jcPtr = new CJoinConfig(CJoinType.INNER, left_column_index, right_column_index,
                                             CJoinAlgorithm.HASH)
            elif join_type == PJoinType.LEFT.value:
                self.jcPtr = new CJoinConfig(CJoinType.LEFT, left_column_index, right_column_index,
                                             CJoinAlgorithm.HASH)
            elif join_type == PJoinType.RIGHT.value:
                self.jcPtr = new CJoinConfig(CJoinType.RIGHT, left_column_index, right_column_index,
                                             CJoinAlgorithm.HASH)
            elif join_type == PJoinType.OUTER.value:
                self.jcPtr = new CJoinConfig(CJoinType.OUTER, left_column_index, right_column_index,
                                             CJoinAlgorithm.HASH)
            else:
                raise ValueError("Unsupported Join Type {}".format(join_type))

        elif join_algorithm == PJoinAlgorithm.SORT.value:

            if join_type == PJoinType.INNER.value:
                self.jcPtr = new CJoinConfig(CJoinType.INNER, left_column_index, right_column_index,
                                             CJoinAlgorithm.SORT)
            elif join_type == PJoinType.LEFT.value:
                self.jcPtr = new CJoinConfig(CJoinType.LEFT, left_column_index, right_column_index,
                                             CJoinAlgorithm.SORT)
            elif join_type == PJoinType.RIGHT.value:
                self.jcPtr = new CJoinConfig(CJoinType.RIGHT, left_column_index, right_column_index,
                                             CJoinAlgorithm.SORT)
            elif join_type == PJoinType.OUTER.value:
                self.jcPtr = new CJoinConfig(CJoinType.OUTER, left_column_index, right_column_index,
                                             CJoinAlgorithm.SORT)
            else:
                raise ValueError("Unsupported Join Type {}".format(join_type))
        else:
            if join_type == PJoinType.INNER.value:
                self.jcPtr = new CJoinConfig(CJoinType.INNER, left_column_index, right_column_index)
            elif join_type == PJoinType.LEFT.value:
                self.jcPtr = new CJoinConfig(CJoinType.LEFT, left_column_index, right_column_index)
            elif join_type == PJoinType.RIGHT.value:
                self.jcPtr = new CJoinConfig(CJoinType.RIGHT, left_column_index, right_column_index)
            elif join_type == PJoinType.OUTER.value:
                self.jcPtr = new CJoinConfig(CJoinType.OUTER, left_column_index, right_column_index)
            else:
                raise ValueError("Unsupported Join Type {}".format(join_type))

    @property
    def join_type(self) -> JoinType:
        self.jtPtr = self.jcPtr.GetType()
        return self.jtPtr

    @property
    def join_algorithm(self) -> JoinAlgorithm:
        self.jaPtr = self.jcPtr.GetAlgorithm()
        return self.jaPtr

    @property
    def left_index(self) -> int:
        return self.jcPtr.GetLeftColumnIdx()

    @property
    def right_index(self)-> int:
        return self.jcPtr.GetRightColumnIdx()