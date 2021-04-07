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
>> pytest -q python/test/test_join.py
"""

from utils import create_df, assert_eq
#from python.test.utils import create_df, assert_eq
import pyarrow as pa
import pandas as pd
import numpy as np
import random

import pycylon as cn
from pycylon import CylonContext, DataFrame


def test_df_joining():
    df_c_1, df_p_1 = create_df([random.sample(range(10, 300), 50),
                                random.sample(range(10, 300), 50),
                                random.sample(range(10, 300), 50)])

    df_c_2, df_p_2 = create_df([random.sample(range(10, 300), 50),
                                random.sample(range(10, 300), 50),
                                random.sample(range(10, 300), 50)])

    def do_join(col):
        df_c_1.set_index(col, inplace=True)
        df_c_2.set_index(col, inplace=True)

        #df_p_1.set_index(col, inplace=True, drop=False)
        #df_p_2.set_index(col, inplace=True, drop=False)
        print(col)
        srt_c = df_c_1.join(on=col, other=df_c_2)
        print(srt_c)
        #print(df_p_1)
        #print(df_p_2)
        srt_p = df_p_1.join(on=col, other=df_p_2, lsuffix="l", rsuffix="r")
        print(srt_p)
        #assert_eq(srt_c, srt_p, sort=True)

    # multi column
    for asc in [True, False]:
        for c1 in range(0, 3):
            # single column
            do_join([c1])
            for c2 in range(0, 3):
                if c1 != c2:
                    do_join([c1, c2])
                for c3 in range(0, 3):
                    if c1 != c2 and c1 != c3 and c2 != c3:
                        do_join([c1, c2, c3])


test_df_joining()
