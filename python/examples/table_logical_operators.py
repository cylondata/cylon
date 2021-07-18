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

from pycylon import Table
from pycylon import CylonContext

ctx: CylonContext = CylonContext(config=None, distributed=False)

data1 = [[False, True, False, True], [True, True, False, False]]
columns1 = ['col-1', 'col-2']

data2 = [[True, True, False, False], [False, True, False, True]]
columns2 = ['col-1', 'col-2']

tb1: Table = Table.from_list(ctx, columns1, data1)
tb2: Table = Table.from_list(ctx, columns2, data2)

print("Table 1")
print(tb1)

print("Table 2")
print(tb2)

tb_or = tb1 | tb2
print("Or")
print(tb_or)

tb_and = tb1 & tb2
print("And")
print(tb_and)

tb_inv = ~tb1
print("Table")
print(tb1)
print("Invert Table")
print(tb_inv)
