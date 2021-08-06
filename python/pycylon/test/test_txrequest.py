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

"""
Run test:

>>> pytest -q python/pycylon/test/test_txrequest.py
"""

from pycylon.net.txrequest import TxRequest
import numpy as np

def test_txrequest():
    target = 10
    length = 8
    header = np.array([1, 2, 3, 4], dtype=np.int32)
    header_length = header.shape[0]
    buf = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
    tx = TxRequest(target, buf, length, header, header_length)

    assert tx.target == target
    assert type(tx.buf) == type(buf) and tx.buf.shape == buf.shape and tx.buf.dtype == buf.dtype
    assert type(tx.header) == type(
        header) and tx.header.shape == header.shape and tx.header.dtype == header.dtype
    assert tx.headerLength == header_length
    assert tx.length == length

    # print("To String")
    # print(tx.to_string(b'double', 32))
