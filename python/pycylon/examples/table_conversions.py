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
import numpy as np

ctx: CylonContext = CylonContext(config=None, distributed=False)
data_dictionary = {'col-1': [1, 2, 3, 4], 'col-2': [5, 6, 7, 8], 'col-3': [9, 10, 11, 12]}
tb: Table = Table.from_pydict(ctx, data_dictionary)

print("Convert to PyArrow Table")
print(tb.to_arrow())

print("Convert to Pandas")
print(tb.to_pandas())

print("Convert to Dictionar")
print(tb.to_pydict())

print("Convert to Numpy")
npy: np.ndarray = tb.to_numpy(order='F', zero_copy_only=True)
print(npy)
print(npy.flags)

npy: np.ndarray = tb.to_numpy(order='C', zero_copy_only=True)
print(npy)
print(npy.flags)
