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

print("Equal Operator")

tb1 = tb['col-1'] == 2

print(tb1)

tb2 = tb == 2

print(tb2)


print("Inequal Operator")

tb3 = tb['col-1'] != 2

print(tb3)

tb4 = tb != 2

print(tb4)

print("Lesser Operator")

tb3 = tb['col-1'] < 2

print(tb3)

tb4 = tb < 2

print(tb4)

print("Lesser equal Operator")

tb3 = tb['col-1'] <= 2

print(tb3)

tb4 = tb <= 2

print(tb4)

print("Greater Operator")

tb3 = tb['col-1'] > 2

print(tb3)

tb4 = tb > 2

print(tb4)

print("Greater equal Operator")

tb3 = tb['col-1'] >= 2

print(tb3)

tb4 = tb >= 2

print(tb4)