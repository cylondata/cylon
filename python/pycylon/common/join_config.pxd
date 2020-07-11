##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##

'''
Join Configurations in TwisterX
'''

cdef extern from "../../../cpp/src/cylon/join/join_config.hpp" namespace "cylon::join::config":
    cdef enum CJoinType "cylon::join::config::JoinType":
        CINNER "cylon::join::config::JoinType::INNER"
        CLEFT "cylon::join::config::JoinType::LEFT"
        CRIGHT "cylon::join::config::JoinType::RIGHT"
        COUTER "cylon::join::config::JoinType::FULL_OUTER"

cdef extern from "../../../cpp/src/cylon/join/join_config.hpp" namespace "cylon::join::config":
    cdef enum CJoinAlgorithm "cylon::join::config::JoinAlgorithm":
        CSORT "cylon::join::config::JoinAlgorithm::SORT"
        CHASH "cylon::join::config::JoinAlgorithm::HASH"


cdef extern from "../../../cpp/src/cylon/join/join_config.hpp" namespace "cylon::join::config":
    cdef cppclass CJoinConfig "cylon::join::config::JoinConfig":
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
