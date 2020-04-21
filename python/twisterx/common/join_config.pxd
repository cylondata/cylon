cdef enum _JoinType:
    INNER = 0
    LEFT = 1
    RIGHT = 2
    OUTER = 3

cdef enum _JoinAlgorithm:
    SORT = 10
    HASH = 11

cdef cppclass _JoinConfig:
    _JoinConfig(_JoinType, _JoinAlgorithm, int, int)
    _JoinType get_join_type()
    _JoinAlgorithm get_join_algorithm()
    int get_left_column()
    int get_right_column()