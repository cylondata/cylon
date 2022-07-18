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
>> pytest -q python/pycylon/test/test_gloo.py
"""

import os
import shutil
import tempfile
from multiprocessing import Process

import pandas as pd
from numpy.random import default_rng

from pycylon import CylonEnv, DataFrame
from pycylon.net.gloo_config import GlooStandaloneConfig

FILE_STORE_PATH = os.path.join(tempfile.gettempdir(), 'gloo')
WORLD_SIZE = 4
ROWS = 5


def run_op(env, r):
    rng = default_rng()
    data1 = rng.integers(0, r, size=(r, 2))
    data2 = rng.integers(0, r, size=(r, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))

    print("Distributed Merge")
    df3 = df1.merge(right=df2, on=[0], env=env)
    print(f'res len {len(df3)}')

    env.finalize()


def run_standalone_gloo(rank, world_size):
    conf = GlooStandaloneConfig(rank, world_size)

    conf.set_file_store_path(FILE_STORE_PATH)

    # distributed join
    env = CylonEnv(config=conf)
    run_op(env, ROWS)


def test_gloo():  # confirms that the code is under main function
    shutil.rmtree(FILE_STORE_PATH, ignore_errors=True)
    os.mkdir(FILE_STORE_PATH)

    procs = []

    # instantiating process with arguments
    for i in range(WORLD_SIZE):
        # print(name)
        proc = Process(target=run_standalone_gloo, args=(i, WORLD_SIZE))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

    for proc in procs:
        assert proc.exitcode == 0
