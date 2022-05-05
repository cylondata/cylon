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


import numpy as np
import pyarrow as pa
import pycylon as cn
from pycylon.data.column import Column
from pycylon import Series


def test_column():
    dtype = cn.int32()
    data = pa.array([1, 2, 3], type=pa.int32())
    col = Column(data)

    assert dtype.type == col.dtype.type
    assert data == col.data


def test_series_with_list():
    ld = [1, 2, 3, 4]
    name = 's1'
    dtype = cn.int64()
    s = Series(name, ld)

    assert s.id == name
    assert s.data == pa.array(ld)
    assert s.dtype.type == dtype.type


def test_series_with_numpy():
    ld = np.array([1, 2, 3, 4], dtype=np.int64)
    id = 's1'
    dtype = cn.int64()
    s = Series(id, ld)

    assert s.id == id
    assert s.data == pa.array(ld)
    assert s.dtype.type == dtype.type


def test_series_with_pyarrow():
    ld = pa.array([1, 2, 3, 4], type=pa.int32())
    name = 's1'
    dtype = cn.int32()
    s = Series(name, ld)

    assert s.id == name
    assert s.data == ld
    assert s.dtype.type == dtype.type
