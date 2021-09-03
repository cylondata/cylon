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

import pyarrow as pa

PYTHON_TYPE_ARROW_TYPE_MAP = {
    float: pa.float32(),
    int: pa.int32(),
    str: pa.string()
}

STR_TYPE_ARROW_TYPE_MAP = {
    'int8': pa.int8(),
    'int16': pa.int16(),
    'int32': pa.int32(),
    'int64': pa.int64(),
    'uint8': pa.uint8(),
    'uint16': pa.uint16(),
    'uint32': pa.uint32(),
    'uint64': pa.uint64(),
    'float32': pa.float32(),
    'float64': pa.float64(),
    'double': pa.float64(),
    'half_float': pa.float16(),
    'string': pa.string(),
    'binary': pa.binary(),
    'bool': pa.bool_(),
    'float': pa.float32(),
    'int': pa.int32(),
    'str': pa.string()
}


def _get_arrow_type_from_python_type(python_type):
    try:
        return PYTHON_TYPE_ARROW_TYPE_MAP[python_type]
    except KeyError:
        return None


def _get_arrow_type_from_str_type(cylon_str_type):
    try:
        return STR_TYPE_ARROW_TYPE_MAP[cylon_str_type]
    except KeyError:
        return None


def get_arrow_type(dtype):
    if isinstance(dtype, str):
        return _get_arrow_type_from_str_type(dtype)
    elif isinstance(dtype, type):
        return _get_arrow_type_from_python_type(dtype)
    else:
        raise ValueError(f'Unsupported dtype {dtype}')
