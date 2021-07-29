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
from typing import List

from pycylon.common.join_config cimport CJoinType
from pycylon.common.join_config cimport CJoinAlgorithm
from pycylon.common.join_config cimport CJoinConfig

cpdef enum JoinType:
    INNER = CJoinType.CINNER
    LEFT = CJoinType.CLEFT
    RIGHT = CJoinType.CRIGHT
    OUTER = CJoinType.COUTER

cpdef enum JoinAlgorithm:
    SORT = CJoinAlgorithm.CSORT
    HASH = CJoinAlgorithm.CHASH

StrToJoinAlgorithm = {
    'sort': CJoinAlgorithm.CSORT,
    'hash': CJoinAlgorithm.CHASH
}

StrToJoinType = {
    'inner': CJoinType.CINNER,
    'left': CJoinType.CLEFT,
    'right': CJoinType.CRIGHT,
    'fullouter': CJoinType.COUTER,
    'outer': CJoinType.COUTER
}

cdef class JoinConfig:
    def __cinit__(self, join_type: str, join_algorithm: str, left_column_index: List[int],
                  right_column_index: List[int], left_table_prefix: str = "",
                  right_table_prefix: str = ""):
        """
        :param join_type: passed as a str from one of the ["inner","left","outer","right"]
        :param join_algorithm: passed as a str from one of the ["sort", "hash"]
        :param left_column_index: passed as a int (currently support joining a single column)
        :param right_column_index: passed as a int (currently support joining a single column)
        :return: None
        """
        if join_type is not None and join_algorithm is not None and left_column_index \
                is not None and right_column_index is not None:
            self._get_join_config(join_type=join_type, join_algorithm=join_algorithm,
                                  left_column_index=left_column_index,
                                  right_column_index=right_column_index,
                                  left_table_prefix=left_table_prefix,
                                  right_table_prefix=right_table_prefix)

    cdef _get_join_config(self, join_type, join_algorithm, left_column_index,
                         right_column_index, left_table_prefix,
                         right_table_prefix):
        cdef CJoinType cjoin_type = StrToJoinType[join_type]
        cdef CJoinAlgorithm cjoin_algo = StrToJoinAlgorithm[join_algorithm]

        self.jcPtr = new CJoinConfig(cjoin_type, left_column_index, right_column_index,
                                         cjoin_algo, left_table_prefix.encode(),
                                         right_table_prefix.encode())

    @property
    def join_type(self) -> JoinType:
        '''
        this an accessible property to the users
        :return: JoinType python object
        '''
        self.jtPtr = self.jcPtr.GetType()
        return self.jtPtr

    @property
    def join_algorithm(self) -> JoinAlgorithm:
        '''
        this an accessible property to the users
        :return: JoinAlgorithm python object
        '''
        self.jaPtr = self.jcPtr.GetAlgorithm()
        return self.jaPtr

    @property
    def left_index(self) -> List[int]:
        """
        this an accessible property to the users
        :return: index of the join column in left table
        """
        cdef vector[int] vec = self.jcPtr.GetLeftColumnIdx()
        res = [x for x in vec]
        return res

    @property
    def right_index(self)-> List[int]:
        """
        this an accessible property to the users
        :return: index of the join column in right table
        """
        cdef vector[int] vec = self.jcPtr.GetRightColumnIdx()
        res = [x for x in vec]
        return res

    @property
    def left_prefix(self) -> str:
        return self.jcPtr.GetLeftTableSuffix().decode()

    @property
    def right_prefix(self) -> str:
        return self.jcPtr.GetRightTableSuffix().decode()
