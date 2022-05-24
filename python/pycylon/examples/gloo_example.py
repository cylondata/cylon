#!/usr/bin/env python

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# import subprocess
#
import os
import random
import tempfile
from multiprocessing import Process

from pycylon import CylonEnv, DataFrame
from pycylon.net.gloo_config import GlooStandaloneConfig

FILE_STORE_PATH = os.path.join(tempfile.gettempdir(), 'gloo')
WORLD_SIZE = 4


def run_op(env):
    df1 = DataFrame([random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5),
                     random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5)])
    df2 = DataFrame([random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5),
                     random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5)])
    print("Distributed Merge")
    df3 = df1.merge(right=df2, on=[0], env=env)
    print(df3)

    env.finalize()


def run_standalone_gloo(rank, world_size):
    conf = GlooStandaloneConfig(rank, world_size)

    if not os.path.isdir(FILE_STORE_PATH):
        os.mkdir(FILE_STORE_PATH)
    conf.set_file_store_path(FILE_STORE_PATH)

    # distributed join
    env = CylonEnv(config=conf)
    run_op(env)


if __name__ == "__main__":  # confirms that the code is under main function
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
