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
>> pytest -q python/test/test_comm_type.py
'''

from pycylon.net.comm_type import CommType

def test_types():
    assert CommType.LOCAL.value == 0
    assert CommType.MPI.value == 1
    assert CommType.TCP.value == 2
    assert CommType.UCX.value == 3