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
Run test
>> pytest -q python/pycylon/test/test_status.py
'''

from pycylon.commons import Code
from pycylon.commons import Status


def test_status():
    msg = b"a"
    code = Code.IOError
    s = Status(code, msg)

    assert (s.get_code() == Code.IOError)
    assert (s.get_msg() == msg.decode())
    assert (s.is_ok() is False)

    msg = b""
    code = Code.IOError
    s = Status(code, msg)

    assert (s.get_code() == Code.IOError)
    assert (len(s.get_msg()) == 0)
    assert (s.is_ok() is False)

    s = Status(Code.OK)
    assert (s.get_code() == Code.OK)
    assert (len(s.get_msg()) == 0)
    assert (s.is_ok())

    s = Status()
    assert (s.get_code() == Code.OK)
    assert (len(s.get_msg()) == 0)
    assert (s.is_ok())
