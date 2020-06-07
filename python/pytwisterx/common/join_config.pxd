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
