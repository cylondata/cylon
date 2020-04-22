cdef extern from "../../../cpp/src/twisterx/join/join_config.h" namespace "twisterx::join::config":
    cdef enum _JoinType "twisterx::join::config::JoinType":
        INNER "twisterx::join::config::JoinType::INNER"
        LEFT "twisterx::join::config::JoinType::LEFT"
        RIGHT "twisterx::join::config::JoinType::RIGHT"
        OUTER "twisterx::join::config::JoinType::FULL_OUTER"

cdef extern from "../../../cpp/src/twisterx/join/join_config.h" namespace "twisterx::join::config":
    cdef enum _JoinAlgorithm "twisterx::join::config::JoinAlgorithm":
        SORT "twisterx::join::config::JoinAlgorithm::SORT"
        HASH "twisterx::join::config::JoinAlgorithm::HASH"

cdef cppclass _JoinConfig:
    _JoinConfig(_JoinType, _JoinAlgorithm, int, int)
    _JoinType get_join_type()
    _JoinAlgorithm get_join_algorithm()
    int get_left_column()
    int get_right_column()