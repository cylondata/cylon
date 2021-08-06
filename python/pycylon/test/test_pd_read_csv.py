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
>> pytest -q python/pycylon/test/test_pd_read_csv.py
'''

import pandas as pd

from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig


def test_pd_read_csv():
    env = CylonEnv(config=MPIConfig())

    df1 = DataFrame(pd.read_csv('/tmp/user_usage_tm_1.csv'))
    df2 = DataFrame(pd.read_csv('/tmp/user_device_tm_1.csv'))

    df1 = df1.set_index([3], drop=True)
    df2 = df2.set_index([0], drop=True)

    df1.to_table().retain_memory(False)
    df2.to_table().retain_memory(False)

    df3 = df1.merge(right=df2, left_on=[3], right_on=[0], algorithm='sort', env=env)

    assert len(df3)
