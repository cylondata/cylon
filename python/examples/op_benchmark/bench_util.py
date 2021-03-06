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

import os
import numpy as np
import pandas as pd


def add_record_to_stats_file(file_path: str, record: str):
    if os.path.exists(file_path):
        with open(file_path, "a+") as fp:
            fp.write(record + "\n")
    else:
        with open(file_path, "w") as fp:
            fp.write(record + "\n")


def get_random_data_column(num_rows: int, unique_factor: float, with_null: bool = False, null_per: float = 0.9,
                           stringify: bool = False):
    if with_null:
        null_row_count = int(num_rows * null_per)
        gen_record_size = int(num_rows * unique_factor)
        null_col = [None] * null_row_count
        data_col = np.random.randint(gen_record_size, size=num_rows - null_row_count).tolist()
        null_col = null_col + data_col
        return null_col
    else:
        gen_record_size = int(num_rows * unique_factor)
        return np.random.randint(gen_record_size, size=num_rows)


def get_dataframe(num_rows: int, num_cols: int, unique_factor: float, with_null: bool = False, null_per: float = 0.9,
                  stringify: bool = False):
    if with_null:
        pdf = pd.DataFrame({'data{}'.format(i): get_random_data_column(num_rows=num_rows,
                                                                       unique_factor=unique_factor,
                                                                       with_null=with_null, null_per=null_per,
                                                                       stringify=stringify)
                            for i in range(num_cols)})
        pdf = pdf.sample(frac=1)
        if stringify:
            return pdf.astype('str')
        return pdf
    else:
        pdf = pd.DataFrame({'data{}'.format(i): get_random_data_column(num_rows=num_rows,
                                                                       unique_factor=unique_factor,
                                                                       stringify=stringify)
                            for i in range(num_cols)})
        if stringify:
            return pdf.astype('str')
        return pdf
