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

from typing import Union
import pandas as pd
import numpy as np

from pycylon import DataFrame


def create_df(data) -> Union[DataFrame, pd.DataFrame]:
    # np.T is a temp fix to address inconsistencies
    return DataFrame(data), pd.DataFrame(np.array(data).T)


def assert_eq(df_c: DataFrame, df_p: pd.DataFrame, sort=False):
    if sort:  # sort by every column
        print(df_c.sort_values(
            by=[*range(0, df_c.shape[1])]).to_numpy(order='F',  zero_copy_only=False))

        print(df_p.to_numpy())
        assert np.array_equal(df_c.sort_values(
            by=[*range(0, df_c.shape[1])]).to_pandas().to_numpy(),  df_p.sort_values(
            by=[*range(0, df_p.shape[1])]).to_numpy())
    else:
        assert np.array_equal(df_c.to_numpy(order='F',  zero_copy_only=False),  df_p.to_numpy())
