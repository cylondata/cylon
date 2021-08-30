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
Run test:
>> pytest -q python/pycylon/test/test_join_config.py
'''

from pycylon.commons import JoinAlgorithm
from pycylon.commons import JoinConfig
from pycylon.commons import JoinType


def test_join_configs():
    assert JoinAlgorithm.SORT.value == 0
    assert JoinAlgorithm.HASH.value == 1
    assert JoinType.INNER.value == 0
    assert JoinType.OUTER.value == 3
    assert JoinType.LEFT.value == 1
    assert JoinType.RIGHT.value == 2

    c = JoinType.LEFT
    a = JoinAlgorithm.HASH

    assert (c == JoinType.LEFT)
    assert (a != JoinAlgorithm.SORT)
    assert (a == JoinAlgorithm.HASH)

    joinconfig = JoinConfig(join_type="left", join_algorithm="hash", left_column_index=[0],
                            right_column_index=[1])

    assert joinconfig.left_index[0] == 0 and joinconfig.right_index[0] == 1

    assert joinconfig.join_type == 1 and joinconfig.join_algorithm == 1


