from pycylon import CylonEnv, DataFrame
from pycylon.distributed_io import read_csv_dist, read_json_dist, read_parquet_dist, write_csv_dist, write_json_dist, write_parquet_dist

from pycylon.net import MPIConfig
import tempfile

"""
Run test:
>> mpirun -n 4 python -m pytest --with-mpi -q python/pycylon/test/test_io.py
"""
import pytest


@pytest.mark.mpi
def test_csv():
    env: CylonEnv = CylonEnv(config=MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    # one file per worker to read
    input_files = "data/mpiops/sales_nulls_nunascii_*.csv"
    df = read_csv_dist(input_files, env=env)

    rows_per_worker = [25, 25, 25, 25]
    assert len(df) == rows_per_worker[env.rank], \
        f'Read CSV DataFrame row count [{len(df)}] does not match given row count [{rows_per_worker[env.rank]}]'

    # write csv files to a temporary directory
    with tempfile.TemporaryDirectory() as dirpath:
        out_file = write_csv_dist(df, dirpath, env)
        df2 = read_csv_dist(paths={env.rank: out_file}, env=env)
        assert len(df) == len(df2), \
            f'Read CSV DataFrame row count [{len(df2)}] does not written dataframe row count [{len(df)}]'

    # only two workers read a file, the others do not read a file
    # they need to get a proper empty DataFrame
    input_files = ["data/mpiops/sales_nulls_nunascii_0.csv", "data/mpiops/sales_nulls_nunascii_1.csv"]
    df = read_csv_dist(input_files, env=env)

    rows_per_worker = [25, 25, 0, 0]
    assert len(df) == rows_per_worker[env.rank], \
        f'Read CSV DataFrame row count [{len(df)}] does not match given row count [{rows_per_worker[env.rank]}]'


@pytest.mark.mpi
def test_json():
    env: CylonEnv = CylonEnv(config=MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    # one file per worker to read
    input_files = "data/json/sales_*.json"
    df = read_json_dist(input_files, env=env, lines=True)

    rows_per_worker = [25, 25, 25, 25]
    assert len(df) == rows_per_worker[env.rank], \
        f'Read JSON DataFrame row count [{len(df)}] does not match given row count [{rows_per_worker[env.rank]}]'

    # write json files to a temporary directory
    with tempfile.TemporaryDirectory() as dirpath:
        out_file = write_json_dist(df, dirpath, env, orient="records", lines=True, force_ascii=False)
        df2 = read_json_dist(paths={env.rank: out_file}, env=env, lines=True)
        assert len(df) == len(df2), \
            f'Read JSON DataFrame row count [{len(df2)}] does not written dataframe row count [{len(df)}]'

    # only two workers read a file, the others do not read a file
    # they need to get a proper empty DataFrame
    input_files = ["data/json/sales_0.json", "data/json/sales_1.json"]
    df = read_json_dist(input_files, env=env, lines=True)

    rows_per_worker = [25, 25, 0, 0]
    assert len(df) == rows_per_worker[env.rank], \
        f'Read CSV DataFrame row count [{len(df)}] does not match given row count [{rows_per_worker[env.rank]}]'

@pytest.mark.mpi
def test_parquet():
    env: CylonEnv = CylonEnv(config=MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    rows_per_worker = [25, 25, 25, 25]
    # reading all files in the directory with parquet extension
    input_files = "data/parquet/*.parquet"
    df = read_parquet_dist(input_files, env=env)
    assert len(df) == rows_per_worker[env.rank], \
        f'Read Parquet DataFrame row count [{len(df)}] does not match the given row count [{rows_per_worker[env.rank]}]'

    # reading all files in the directory by using _metadata file
    input_dir = "data/parquet"
    df = read_parquet_dist(input_dir, env=env)
    assert len(df) == rows_per_worker[env.rank], \
        f'Read Parquet DataFrame row count [{len(df)}] does not match the given row count [{rows_per_worker[env.rank]}]'

    # write parquet files to a temporary directory with _metadata
    with tempfile.TemporaryDirectory() as dirpath:
        out_file = write_parquet_dist(df, dirpath, env=env)
        df2 = read_parquet_dist(paths={env.rank: out_file}, env=env)
        assert len(df) == len(df2), \
            f'Read Parquet DataFrame row count [{len(df2)}] does not match the written dataframe row count [{len(df)}]'

    # only two workers read a file, the others do not read a file
    # they need to get a proper empty DataFrame
    input_files = ["data/parquet/part_0000.parquet", "data/parquet/part_0001.parquet"]
    df = read_parquet_dist(input_files, env=env)
    rows_per_worker = [25, 25, 0, 0]
    assert len(df) == rows_per_worker[env.rank], \
        f'Read Parquet DataFrame row count [{len(df)}] does not match the given row count [{rows_per_worker[env.rank]}]'

    # only two workers read a file with ranks specified, the others do not read a file
    # they need to get a proper empty DataFrame
    input_files = {1: "data/parquet/part_0001.parquet", 2: "data/parquet/part_0003.parquet"}
    df = read_parquet_dist(input_files, env=env)
    rows_per_worker = [0, 25, 25, 0]
    assert len(df) == rows_per_worker[env.rank], \
        f'Read Parquet DataFrame row count [{len(df)}] does not match the given row count [{rows_per_worker[env.rank]}]'
