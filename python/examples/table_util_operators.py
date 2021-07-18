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

data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
columns = ['col-1', 'col-2', 'col-3']

tb: Table = Table.from_list(ctx, columns, data)
print("Full Dataset")
print(tb)

tb.rename({'col-1': 'col_1'})

print(tb)

tb.rename(['c1', 'c2', 'c3'])

print(tb)

print(tb.add_prefix('old_'))

print(tb.index)

tb.set_index(['a', 'b', 'c', 'd'])

print(tb.index)

print(tb.index.index_values)

data1 = [[1, None, 3, 4], [5, 6, 7, 8], [9, 10, 11, None]]
columns1 = ['col-1', 'col-2', 'col-3']

tb_na: Table = Table.from_list(ctx, columns1, data1)
print("DropNa")
print(tb_na)
# axis = 0
tb_na_new = tb_na.dropna(how='any')

print(tb_na_new)

tb_na_new = tb_na.dropna(how='all')

print(tb_na_new)

# axis = 1

tb_na_new = tb_na.dropna(axis=1, how='any')

print(tb_na_new)

print(tb_na)

tb_na_new = tb_na.dropna(axis=1, how='all')

print(tb_na_new)

tb_na.dropna(axis=1, how='any', inplace=True)

print(tb_na)
