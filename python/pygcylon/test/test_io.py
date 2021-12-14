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
import pyarrow
import pytest
import cudf
import pycylon as cy
import pygcylon as gcy
import tempfile
import pyarrow.parquet as pq
from pycylon.net.comm_ops import bcast_buffer


@pytest.mark.mpi
def test_csv():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    # one file per worker to read
    input_files = "data/mpiops/sales_nulls_nunascii_*.csv"
    df = gcy.read_csv(input_files, env=env)

    rows_per_worker = [25, 25, 25, 25]
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
    df = gcy.read_csv(input_files, env=env)

    rows_per_worker = [25, 25, 0, 0]
    assert len(df) == rows_per_worker[env.rank], \
        f'Read CSV DataFrame row count [{len(df)}] does not match given row count [{rows_per_worker[env.rank]}]'


@pytest.mark.mpi
def test_json():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    # one file per worker to read
    input_files = "data/json/sales_*.json"
    df = gcy.read_json(input_files, env=env, lines=True)

    rows_per_worker = [25, 25, 25, 25]
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
    df = gcy.read_json(input_files, env=env, lines=True)

    rows_per_worker = [25, 25, 0, 0]
    assert len(df) == rows_per_worker[env.rank], \
        f'Read CSV DataFrame row count [{len(df)}] does not match given row count [{rows_per_worker[env.rank]}]'


@pytest.mark.mpi
def test_parquet():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    rows_per_worker = [25, 25, 25, 25]
    # reading all files in the directory with parquet extension
    input_files = "data/parquet/*.parquet"
    df = gcy.read_parquet(input_files, env=env)
    assert len(df) == rows_per_worker[env.rank], \
        f'Read Parquet DataFrame row count [{len(df)}] does not match the given row count [{rows_per_worker[env.rank]}]'

    # reading all files in the directory by using _metadata file
    input_dir = "data/parquet"
    df = gcy.read_parquet(input_dir, env=env)
    assert len(df) == rows_per_worker[env.rank], \
        f'Read Parquet DataFrame row count [{len(df)}] does not match the given row count [{rows_per_worker[env.rank]}]'

    # write parquet files to a temporary directory with _metadata
    with tempfile.TemporaryDirectory() as dirpath:
        out_file = df.to_parquet(dirpath, env=env)
        df2 = gcy.read_parquet(paths={env.rank: out_file}, env=env)
        assert len(df) == len(df2), \
            f'Read Parquet DataFrame row count [{len(df2)}] does not match the written dataframe row count [{len(df)}]'

    # only two workers read a file, the others do not read a file
    # they need to get a proper empty DataFrame
    input_files = ["data/parquet/part_0000.parquet", "data/parquet/part_0001.parquet"]
    df = gcy.read_parquet(input_files, env=env)
    rows_per_worker = [25, 25, 0, 0]
    assert len(df) == rows_per_worker[env.rank], \
        f'Read Parquet DataFrame row count [{len(df)}] does not match the given row count [{rows_per_worker[env.rank]}]'

    # only two workers read a file with ranks specified, the others do not read a file
    # they need to get a proper empty DataFrame
    input_files = {1: "data/parquet/part_0001.parquet", 2: "data/parquet/part_0003.parquet"}
    df = gcy.read_parquet(input_files, env=env)
    rows_per_worker = [0, 25, 25, 0]
    assert len(df) == rows_per_worker[env.rank], \
        f'Read Parquet DataFrame row count [{len(df)}] does not match the given row count [{rows_per_worker[env.rank]}]'


@pytest.mark.mpi
def test_bcast():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    # read the same parquet file schema by all workers
    input_file = "data/parquet/part_0000.parquet"
    schema = pq.read_schema(input_file)

    # get the schema of the first worker and broadcast it to all other workers
    buf = None
    bcast_root = 0
    if env.rank == bcast_root:
        buf = schema.serialize()

    buf = bcast_buffer(buf, bcast_root, env.context)

    # deserialize the schema
    with pyarrow.ipc.open_stream(buf) as reader:
        received_schema = reader.schema

    assert schema.equals(received_schema), \
        f'Broadcasted schema and the original schema are not the same. Worker rank: {[env.rank]}'


#    env.finalize()
