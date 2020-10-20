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

from pycylon.commons import Code
from pycylon.commons import Status

#print(Code.OK.value, Code.KeyError.value, Code.IOError.value, Code.AlreadyExists.value)

# Case I
# Input From Abstract Python API: code = -1, msg = None and Code = None
msg = b"a"
code = -1
_code = Code.IOError
s = Status(code, msg, _code)

assert (s.get_code() == Code.IOError)
assert (s.get_msg() == msg)
assert (s.is_ok() == False)

# Case II
# Input From Abstract Python API: code = 1, msg = None and Code = None
msg = b""
code = 1
_code = -1
s = Status(code, msg, _code)

assert (s.get_code() == 1)
assert (s.get_msg() == msg)
assert (s.is_ok() == False)

# Case III
# Input From Abstract Python API: code = -1, msg = "a" and Code = Code.OK

msg = b"a"
code = -1
_code = Code.OK
s = Status(code, msg, _code)

assert (s.get_code() == Code.OK)
assert (s.get_msg() == msg)
assert (s.is_ok() == True)

# Case IV
# Input From Abstract Python API: code = 0, msg = "a" and Code = None
msg = b"a"
code = 0
_code = -1
s = Status(code, msg, _code)

assert (s.get_code() == code)
assert (s.get_msg() == msg)
assert (s.is_ok() == True)

# Case V
# Input From Abstract Python API: code = 1, msg = None and Code = None
msg = b""
code = -1
_code = Code.IOError
s = Status(code, msg, _code)

assert (s.get_code() == Code.IOError)
assert (s.get_msg() == msg)
assert (s.is_ok() == False)