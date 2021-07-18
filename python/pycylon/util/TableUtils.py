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

def resolve_column_index_from_column_name(column_name, table) -> int:
    index = None
    for idx, col_name in enumerate(table.column_names):
        if column_name == col_name:
            return idx
    if index is None:
        raise ValueError(f"Column {column_name} does not exist in the table")