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
>> pytest -q python/test/test_data_types.py
'''

from pycylon.data.data_type import Type
from pycylon.data.data_type import Layout

def test_data_types_1():

    # Here just check some types randomly
    assert Type.BOOL.value == 0
    assert Layout.FIXED_WIDTH.value == 1

    assert Type.INT32 == 6
    assert Layout.FIXED_WIDTH == 1










