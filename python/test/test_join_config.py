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

from pycylon.commons import JoinAlgorithm
from pycylon.commons import JoinConfig
from pycylon.commons import JoinType

# print(JoinAlgorithm.SORT.value)
# print(JoinAlgorithm.HASH.value)
# print(JoinType.INNER.value)
# print(JoinType.OUTER.value)
# print(JoinType.LEFT.value)
# print(JoinType.RIGHT.value)


c = JoinType.LEFT
a = JoinAlgorithm.HASH

assert (c == JoinType.LEFT)
assert (a != JoinAlgorithm.SORT)
assert (a == JoinAlgorithm.HASH)


joinconfig = JoinConfig(join_type="left", join_algorithm="hash", left_column_index=0, right_column_index=1)

print(joinconfig.left_index, joinconfig.right_index)

print(joinconfig.join_type, joinconfig.join_algorithm)


