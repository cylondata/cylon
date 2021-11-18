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
running test case
>>  mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi -q python/pygcylon/test/test_io.py
'''

import pytest
import cudf
import pycylon as cy
import pygcylon as gcy
import tempfile

@pytest.mark.mpi
def test_csv():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    # one file per worker to read
    input_files = "data/mpiops/sales_nulls_nunascii_*.csv"
    rows_per_worker = [25, 25, 25, 25]
    df = gcy.read_csv(input_files, env=env)
    assert len(df) == rows_per_worker[env.rank], \
        f'Read CSV DataFrame row count [{len(df)}] does not match given row count [{rows_per_worker[env.rank]}]'

    # write csv files to a temporary directory
    with tempfile.TemporaryDirectory() as dirpath:
        out_file = df.to_csv(dirpath, env=env)
        df2 = gcy.read_csv(paths={env.rank: out_file}, env=env)
        assert len(df) == len(df2), \
            f'Read CSV DataFrame row count [{len(df2)}] does not written dataframe row count [{len(df)}]'

    # only two workers read a file, the others do not read a file
    # they need to get a proper empty DataFrame
    input_files = ["data/mpiops/sales_nulls_nunascii_0.csv", "data/mpiops/sales_nulls_nunascii_1.csv"]
    rows_per_worker = [25, 25, 0, 0]
    df = gcy.read_csv(input_files, env=env)
    assert len(df) == rows_per_worker[env.rank], \
        f'Read CSV DataFrame row count [{len(df)}] does not match given row count [{rows_per_worker[env.rank]}]'


@pytest.mark.mpi
def test_json():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    # one file per worker to read
    input_files = "data/json/sales_*.json"
    rows_per_worker = [25, 25, 25, 25]
    df = gcy.read_json(input_files, env=env, lines=True)
    assert len(df) == rows_per_worker[env.rank], \
        f'Read JSON DataFrame row count [{len(df)}] does not match given row count [{rows_per_worker[env.rank]}]'

    # write json files to a temporary directory
    with tempfile.TemporaryDirectory() as dirpath:
        out_file = df.to_json(dirpath, env=env, orient="records", lines=True, force_ascii=False)
        df2 = gcy.read_json(paths={env.rank: out_file}, env=env, lines=True)
        assert len(df) == len(df2), \
            f'Read JSON DataFrame row count [{len(df2)}] does not written dataframe row count [{len(df)}]'

    # only two workers read a file, the others do not read a file
    # they need to get a proper empty DataFrame
    input_files = ["data/json/sales_0.json", "data/json/sales_1.json"]
    rows_per_worker = [25, 25, 0, 0]
    df = gcy.read_json(input_files, env=env, lines=True)
    assert len(df) == rows_per_worker[env.rank], \
        f'Read CSV DataFrame row count [{len(df)}] does not match given row count [{rows_per_worker[env.rank]}]'


#    env.finalize()
