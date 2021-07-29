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

from typing import List
import pandas as pd


def rename_with_new_column_names(df: pd.DataFrame, new_columns: List[str]):
    old_names = df.columns.array
    rename_map = {}
    for old_name, new_name in zip(old_names, new_columns):
        rename_map[old_name] = new_name
    df.rename(columns=rename_map, inplace=True)
    return df