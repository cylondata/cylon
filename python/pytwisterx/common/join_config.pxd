cdef extern from "../../../cpp/src/twisterx/join/join_config.h" namespace "twisterx::join::config":
    cdef enum CJoinType "twisterx::join::config::JoinType":
        CINNER "twisterx::join::config::JoinType::INNER"
        CLEFT "twisterx::join::config::JoinType::LEFT"
        CRIGHT "twisterx::join::config::JoinType::RIGHT"
        COUTER "twisterx::join::config::JoinType::FULL_OUTER"

cdef extern from "../../../cpp/src/twisterx/join/join_config.h" namespace "twisterx::join::config":
    cdef enum CJoinAlgorithm "twisterx::join::config::JoinAlgorithm":
        CSORT "twisterx::join::config::JoinAlgorithm::SORT"
        CHASH "twisterx::join::config::JoinAlgorithm::HASH"


cdef extern from "../../../cpp/src/twisterx/join/join_config.h" namespace "twisterx::join::config":
    cdef cppclass CJoinConfig "twisterx::join::config::JoinConfig":
        CJoinConfig(CJoinType type, int, int)
        CJoinConfig(CJoinType, int, int, CJoinAlgorithm)
        CJoinConfig InnerJoin(int, int)
        CJoinConfig LeftJoin(int, int)
        CJoinConfig RightJoin(int, int)
        CJoinConfig FullOuterJoin(int, int)
        CJoinConfig InnerJoin(int, int, CJoinAlgorithm)
        CJoinConfig LeftJoin(int, int, CJoinAlgorithm)
        CJoinConfig RightJoin(int, int, CJoinAlgorithm)
        CJoinConfig FullOuterJoin(int, int, CJoinAlgorithm)
        CJoinType GetType()
        CJoinAlgorithm GetAlgorithm()
        int GetLeftColumnIdx()
        int GetRightColumnIdx()
