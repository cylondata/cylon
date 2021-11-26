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

from __future__ import annotations
from typing import Hashable, List, Tuple, Dict, Optional, Sequence, Union, Iterable

import glob
import os

import cudf
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
import pygcylon as gcy
from pycylon.frame import CylonEnv
from pycylon.net.comm_ops import allgather_buffer, gather_buffer


def _get_files(paths: Union[str, List[str]], ext: str = None) -> List[str]:
    """
    convert a list of glob strings to a list of files
    if directory paths are given, ext string added to the directory path.
    Possible ext values are: "*.parquet", "*.csv", "*.json"

    get all file names as a single list
    sort them alphabetically and return
    """
    if isinstance(paths, str):
        paths = [paths]

    all_files = []
    for p in paths:
        p = os.path.join(p, ext) if (os.path.isdir(p) and ext is not None) else p
        all_files.extend(glob.glob(p))

    return sorted(all_files)


def _indices_per_worker(nfiles, my_rank, nworkers):
    """
    calculate indices for a worker by evenly dividing files among them
    """
    files_per_worker = int(nfiles / nworkers)
    workers_with_exra_file = nfiles % nworkers
    start = files_per_worker * my_rank + \
            (my_rank if my_rank < workers_with_exra_file else workers_with_exra_file)
    end = start + files_per_worker + (1 if my_rank < workers_with_exra_file else 0)
    return start, end


def _get_worker_files(paths: List[str], env: CylonEnv, ext: str = None) -> List[str]:
    """
    calculate the list of files that this worker will read
    """
    all_files = _get_files(paths=paths, ext=ext)
    start, end = _indices_per_worker(len(all_files), env.rank, env.world_size)
    return all_files[start:end]


def _all_schemas_equal(schema: pyarrow.Schema, env: CylonEnv) -> pyarrow.Schema:
    """
    after reading files, check whether all DataFrame schemas are equal
    return a schema in case the worker has no DataFrame and needs to create an empty table
    """

    # serialize the schema if exist, otherwise allocate empty buffer
    buf = schema.serialize() if schema is not None else pa.allocate_buffer(0)

    buffers = allgather_buffer(buf=buf, context=env.context)

    schemas = []
    for schema_buf in buffers:
        if len(schema_buf) > 0:
            with pa.ipc.open_stream(schema_buf) as reader:
                schemas.append(reader.schema)

    if len(schemas) == 0:
        raise ValueError("No worker has any schema.")

    # make sure all schemas are equal, compare consecutive ones
    for first, second in zip(schemas, schemas[1:]):
        if not first.equals(second):
            raise ValueError("Not all DataFrame schemas are equal.")

    return schemas[0]


def _gen_file_name(rank: int, file_ext: str) -> str:
    """
    generate filename to write a DataFrame partition
    """
    if rank < 10:
        numeric_ext = "000" + str(rank)
    elif rank < 100:
        numeric_ext = "00" + str(rank)
    elif rank < 1000:
        numeric_ext = "0" + str(rank)
    else:
        numeric_ext = str(rank)
    return "part_" + numeric_ext + "." + file_ext


def _determine_file_name_to_write(dir_path, name_function, file_ext, env):
    """
    Determine the filename to write for a worker
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_name = _gen_file_name(env.rank, file_ext) if name_function is None else name_function(env.rank)
    return os.path.join(dir_path, file_name)


def _read_csv_or_json(read_fun, paths, env, **kwargs) -> gcy.DataFrame:
    """
    Read CSV or JSON files and construct a distributed DataFrame
    """

    if isinstance(paths, str):
        paths = [paths]

    if isinstance(paths, list) and all(isinstance(p, str) for p in paths):
        worker_files = _get_worker_files(paths, env)
    elif isinstance(paths, dict) and \
            all(isinstance(key, int) and (isinstance(val, list) or isinstance(val, str)) for key, val in paths.items()):
        worker_files = _get_files(paths[env.rank]) if env.rank in paths else []
    else:
        raise ValueError("paths must be: Union[str, List[str], Dict[int, Union[str, List[str]]]]")

    cdf = None
    # if there is only one file to read
    if len(worker_files) == 1:
        cdf = read_fun(worker_files[0], **kwargs)
    elif len(worker_files) > 1:
        cdf_list = []
        for worker_file in worker_files:
            cdf_list.append(read_fun(worker_file, **kwargs))
        cdf = cudf.concat(cdf_list)

    schema = cdf.head(0).to_arrow().schema if cdf is not None else None
    df_schema = _all_schemas_equal(schema, env=env)
    if cdf is None:
        cdf = cudf.DataFrame.from_arrow(df_schema.empty_table())

    return gcy.DataFrame.from_cudf(cdf)


def read_csv(paths: Union[str, List[str], Dict[int, Union[str, List[str]]]],
             env: CylonEnv,
             **kwargs) -> gcy.DataFrame:
    """
    Read CSV files and construct a distributed DataFrame
    We try to distribute files among workers evenly.
    When a worker reads multiple files, it concatenates them and generates a single DataFrame

    All files have to have the same schema.
    Otherwise an error will be thrown.

    All path strings are evaluated as glob strings.

    If a dictionary is provided as the "paths" parameter,
      workers with given ranks read those matching files and concatenate them.
    If a string or list of strings are provided, first the list of all files to read
      are determined by evaluating glob paths.
      Then files are sorted alphabetically and divided among the workers evenly.

    Examples:
        # read all csv files in a directory
        gcy.read_csv(paths="/pat/to/dir/*.csv", env=env)

        # read only a single csv file
        gcy.read_csv(paths="/pat/to/dir/example.csv", env=env)

        # read only a list of csv files
        gcy.read_csv(paths=["/pat/to/dir/example_0.csv", "/pat/to/dir/example_1.csv"], env=env)

        # specify the files to be read by individual workers
        gcy.read_csv(paths={0: "/pat/to/dir/example_0.csv", 1: "/pat/to/dir/example_1.csv"}, env=env)
        gcy.read_csv(paths={0: "/pat/to/dir/example_0*.csv", 1: "/pat/to/dir/example_1*.csv"}, env=env)

        # specify many files to be read by individual workers
        gcy.read_csv(paths={0: ["/pat/to/dir/example_0.csv", "/pat/to/dir/example_1*.csv"],
                            1: "/pat/to/dir/example_2*.csv"}, env=env)

    Parameters
    ----------
    paths: Input CSV file paths as glob strings.
           A string, a list of strings, a dictionary of worker ranks to paths
           that can be either a string or a list of strings.
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to cudf.read_csv function

    Returns
    -------
    A new distributed DataFrame constructed by reading all CSV files
    """
    return _read_csv_or_json(cudf.read_csv, paths=paths, env=env, **kwargs)


def write_csv(df: gcy.DataFrame,
              dir_path: str,
              env: CylonEnv,
              name_function: callable = None,
              **kwargs) -> str:
    """
    Write DataFrames to CSV files

    Each worker writes a single CSV file.
    All files are written to the given directory.
    If the name_function is not provided:
      each worker creates the output file with the pattern: "part_<rank>.csv"
    If the name_function parameter is given:
      this function is used to generate the output filename by each worker.
      this function must take an int as the argument and return a string as the filename.
      each worker calls this function with its worker rank.

    Parameters
    ----------
    dir_path: Output directory for CSV files.
    env: CylonEnv object for this DataFrame
    name_function: a function to create the filename for that worker
    kwargs: the parameters that will be passed on to cudf.DataFrame.to_csv function

    Returns
    -------
    Filename written
    """
    outfile = _determine_file_name_to_write(dir_path=dir_path, name_function=name_function, file_ext="csv", env=env)
    df.to_cudf().to_csv(outfile, **kwargs)
    return outfile


def read_json(paths: Union[str, List[str], Dict[int, Union[str, List[str]]]],
              env: CylonEnv,
              **kwargs) -> gcy.DataFrame:
    """
    Read JSON files and construct a distributed DataFrame
    We try to distribute files among workers evenly.
    When a worker reads multiple files, it concatenates them and generates a single DataFrame

    All files have to have the same schema.
    Otherwise an error will be thrown.

    All path strings are evaluated as glob strings.

    If a dictionary is provided as the "paths" parameter,
      workers with given ranks read those matching files and concatenate them.
    If a string or list of strings are provided, first the list of all files to read
      are determined by evaluating glob paths.
      Then files are sorted alphabetically and divided among the workers evenly.

    Examples:
        # read all json files in a directory
        gcy.read_json(paths="/pat/to/dir/*.json", env=env)

        # read only a single json file
        gcy.read_json(paths="/pat/to/dir/example.json", env=env)

        # read only a list of json files
        gcy.read_json(paths=["/pat/to/dir/example_0.json", "/pat/to/dir/example_1.json"], env=env)

        # specify the files to be read by individual workers
        gcy.read_json(paths={0: "/pat/to/dir/example_0.json", 1: "/pat/to/dir/example_1.json"}, env=env)
        gcy.read_json(paths={0: "/pat/to/dir/example_0*.json", 1: "/pat/to/dir/example_1*.json"}, env=env)

        # specify many files to be read by individual workers
        gcy.read_json(paths={0: ["/pat/to/dir/example_0.json", "/pat/to/dir/example_1*.json"],
                            1: "/pat/to/dir/example_2*.json"}, env=env)

    Parameters
    ----------
    paths: Input JSON file paths as glob strings.
           A string, a list of strings, a dictionary of worker ranks to paths
           that can be either a string or a list of strings.
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to cudf.read_json function

    Returns
    -------
    A new distributed DataFrame constructed by reading all JSON files
    """
    return _read_csv_or_json(cudf.read_json, paths=paths, env=env, **kwargs)


def write_json(df: gcy.DataFrame,
               dir_path: str,
               env: CylonEnv,
               name_function: callable = None,
               **kwargs) -> str:
    """
    Write DataFrames to JSON files

    Each worker writes a single JSON file.
    All files are written to the given directory.
    If the name_function is not provided:
      each worker creates the output file with the pattern: "part_<rank>.json"
    If the name_function parameter is given:
      this function is used to generate the output filename by each worker.
      this function must take an int as the argument and return a string as the filename.
      each worker calls this function with its worker rank.

    Parameters
    ----------
    dir_path: Output directory for JSON files.
    env: CylonEnv object for this DataFrame
    name_function: a function to create the filename for that worker
    kwargs: the parameters that will be passed on to cudf.DataFrame.to_json function

    Returns
    -------
    Filename written
    """
    outfile = _determine_file_name_to_write(dir_path=dir_path, name_function=name_function, file_ext="json", env=env)
    df.to_cudf().to_json(outfile, **kwargs)
    return outfile


#####################################################################################
# Parquet io functions
#####################################################################################


def _serialize_metadata_file(fmd: pa._parquet.FileMetaData) -> pa.Buffer:
    """
    serialize a FileMetaData object
    return empty buffer if None
    """
    if fmd is None:
        return pa.allocate_buffer(0)
    else:
        _, buf_tuple = fmd.__reduce__()
        return buf_tuple[0]


def _deserialize_metadata_buffers(buffers: List[pa.Buffer]) -> pa._parquet.FileMetaData:
    """
    deserialize FileMetaData buffers
    """
    fmd_all = []
    for received_buf in buffers:
        if received_buf.size > 0:
            fmd_all.append(pa._parquet._reconstruct_filemetadata(received_buf))
    return fmd_all


def _schemas_equal(fmd_list: List[pa._parquet.FileMetaData], my_rank: int):
    """
    compare schemas of consecutive FileMetaData objects
    """
    for fmd1, fmd2 in zip(fmd_list, fmd_list[1:]):
        if not fmd1.schema.equals(fmd2.schema):
            f1 = fmd1.row_group(0).column(0).file_path
            f2 = fmd2.row_group(0).column(0).file_path
            raise ValueError(my_rank, f"schemas are not equal for the files: {f1} and {f2}")


def _all_gather_metadata(worker_files: List[str], env: CylonEnv) -> pa._parquet.FileMetaData:
    """
    allgather FileMetaData objects and check schemas for equality
    """
    fmd_list = []
    for worker_file in worker_files:
        fmd = pq.read_metadata(worker_file)
        fmd.set_file_path(worker_file)
        fmd_list.append(fmd)

    # make sure all schemas are equal for this worker
    _schemas_equal(fmd_list, env.rank)

    fmd_single = fmd_list[0] if fmd_list else None
    for fmd in fmd_list[1:]:
        fmd_single.append_row_groups(fmd)

    # serialize and all gather
    buf = _serialize_metadata_file(fmd_single)
    buffers = allgather_buffer(buf=buf, context=env.context)

    # deserialize
    fmd_all = _deserialize_metadata_buffers(buffers)
    if len(fmd_all) == 0:
        raise ValueError("No worker has any parquet files.")

    # make sure all received schemas are equal
    _schemas_equal(fmd_all, env.rank)

    # combine all received schemas into one
    fmd_single = fmd_all[0]
    [fmd_single.append_row_groups(fmd) for fmd in fmd_all[1:]]
    return fmd_single


def _row_groups_this_worker(df: pd.DataFrame, env: CylonEnv):
    """
    Determine the row groups this worker will read.
    Try to distribute rows evenly among workers.
    Each row_group will be read once by a single worker.
    Each worker will get consecutive row_groups in files so that reading might be efficient.
    """
    # if the number of workers are more than the number of row_groups,
    # just return a single row_group for each worker corresponding to its rank or an empty df
    if len(df.index) <= env.world_size:
        return df[env.rank:(env.rank + 1)]

    total_rows = df['row_group_size'].sum()
    remaining_workers = env.world_size
    rows_per_worker = total_rows / remaining_workers
    start_index = 0
    end_index = 0
    last_worker = env.rank == env.world_size - 1
    loop_count = env.rank if last_worker else env.rank + 1
    for _ in range(loop_count):
        rows_this_worker = 0
        start_index = end_index
        for i, rg_size in zip(df.index[start_index:], df['row_group_size'][start_index:]):
            rows_this_worker += rg_size
            if rows_this_worker >= rows_per_worker:
                end_index = i + 1
                break
        total_rows -= rows_this_worker
        remaining_workers -= 1
        rows_per_worker = total_rows / remaining_workers

    return df[end_index:] if last_worker else df[start_index:end_index]


def _construct_df(fmd: pa._parquet.FileMetaData) -> pd.DataFrame:
    """
    construct a DataFrame with three columns
    for calculating row_groups per worker
    """
    row_group_indices = []
    row_group_sizes = []
    files = []
    rg_index = 0
    for i in range(fmd.num_row_groups):
        rg = fmd.row_group(i)
        worker_file = rg.column(0).file_path
        row_group_sizes.append(rg.num_rows)
        if rg_index > 0 and worker_file != files[len(files) - 1]:
            rg_index = 0
        row_group_indices.append(rg_index)
        rg_index += 1
        files.append(worker_file)

    return pd.DataFrame({'row_group_index': row_group_indices, 'row_group_size': row_group_sizes, 'file': files})


def read_parquet(paths: Union[str, List[str], Dict[int, Union[str, List[str]]]],
                 env: CylonEnv,
                 **kwargs) -> gcy.DataFrame:
    """
    Read Parquet files and construct a distributed DataFrame.
    Each row_group is read by exactly one worker.
    Each worker is assigned only consecutive row_groups.
    We try to distribute row_groups among the workers
      to make the row_counts of each worker DataFrame as close as possible.
    When a worker reads data from multiple files,
      it concatenates them and generates a single DataFrame

    All files have to have the same schema.
    Otherwise an error will be thrown.

    All path strings are evaluated as glob strings.

    If a directory is provided as the "paths" parameter,
      we check for the "_metadata" file, if it exists,
        all workers read that metadata file and calculate the row_groups they will read.
      if there is no "_metadata" file in the directory,
        we read all parquet files in that directory.
    If a dictionary is provided as the "paths" parameter,
      workers with given ranks read those matching files and concatenate them.
    If a string or list of strings are provided, first the list of all files to read
      are determined by evaluating glob paths.
      Then the metadata of all files are read,
      row_groups are distributed among the workers to make the num_rows of each worker DataFrame as close as possible.

    Examples:
        # read all parquet files in a directory
        gcy.read_parquet(paths="/pat/to/dir/", env=env)

        # read all parquet files in a directory
        gcy.read_parquet(paths="/pat/to/dir/*.parquet", env=env)

        # read only a single parquet file
        gcy.read_parquet(paths="/pat/to/dir/example.parquet", env=env)

        # read only a list of parquet files
        gcy.read_parquet(paths=["/pat/to/dir/example_0.parquet", "/pat/to/dir/example_1.parquet"], env=env)

        # specify the files to be read by individual workers
        gcy.read_parquet(paths={0: "/pat/to/dir/example_0.parquet", 1: "/pat/to/dir/example_1.parquet"}, env=env)
        gcy.read_parquet(paths={0: "/pat/to/dir/example_0*.parquet", 1: "/pat/to/dir/example_1*.parquet"}, env=env)

        # specify many files to be read by individual workers
        gcy.read_parquet(paths={0: ["/pat/to/dir/example_0.parquet", "/pat/to/dir/example_1*.parquet"],
                            1: "/pat/to/dir/example_2*.parquet"}, env=env)

    Parameters
    ----------
    paths: Input parquet file paths as glob strings.
           A string, a list of strings, a dictionary of worker ranks to paths
           that can be either a string or a list of strings.
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to cudf.read_parquet function

    Returns
    -------
    A new distributed DataFrame constructed by reading all parquet files
    """

    fmd = None
    if isinstance(paths, str):
        if os.path.isdir(paths):
            mdfile = os.path.join(paths, "_metadata")
            if os.path.isfile(mdfile):
                # all workers read the metadata file.
                # one worker may read and broadcast the metadata file.
                # since the metadata file is small, not sure this is needed
                fmd = pq.read_metadata(mdfile)
        paths = [paths]

    if fmd is None:
        if isinstance(paths, list) and all(isinstance(p, str) for p in paths):
            worker_files = _get_worker_files(paths, env, ext="*.parquet")
            file_mappings_given = False
        elif isinstance(paths, dict) and \
                all(isinstance(key, int) and (isinstance(val, list) or isinstance(val, str)) for key, val in paths.items()):
            worker_files = _get_files(paths[env.rank], ext="*.parquet") if env.rank in paths else []
            file_mappings_given = True
        else:
            raise ValueError("paths must be: Union[str, List[str], Dict[int, Union[str, List[str]]]]")

        fmd = _all_gather_metadata(worker_files=worker_files, env=env)
        if file_mappings_given:
            cdf = cudf.read_parquet(worker_files, **kwargs) if worker_files else \
                cudf.DataFrame.from_arrow(fmd.schema.to_arrow_schema().empty_table())
            return gcy.DataFrame.from_cudf(cdf)

    # construct a DataFrame with relevant columns
    pdf = _construct_df(fmd)

    # distribute row_groups to workers
    my_rg_df = _row_groups_this_worker(pdf, env)
    # outfile = "row_groups_" + str(env.rank) + ".csv"
    # my_rg_df.to_csv(outfile)
    # print(env.rank, "written to the file: ", outfile)

    my_files = my_rg_df["file"].unique().tolist()
    my_row_groups = []
    for f in my_files:
        rg_per_file = my_rg_df[my_rg_df['file'] == f]["row_group_index"].tolist()
        my_row_groups.append(rg_per_file)

    if my_files:
        cdf = cudf.read_parquet(my_files, row_groups=my_row_groups, **kwargs)
    else:
        cdf = cudf.DataFrame.from_arrow(fmd.schema.to_arrow_schema().empty_table())

    return gcy.DataFrame.from_cudf(cdf)


def write_parquet(df: gcy.DataFrame,
                  dir_path: str,
                  env: CylonEnv,
                  name_function: callable = None,
                  write_metadata_file: bool = True,
                  **kwargs) -> str:
    """
    Write DataFrames to Parquet files

    Each worker writes a single parquet file.
    All files are written to the given directory.
    If the name_function is not provided:
      each worker creates the output file with the pattern: "part_<rank>.parquet"
    If the name_function parameter is given:
      this function is used to generate the output filename by each worker.
      this function must take an int as the argument and return a string as the filename.
      each worker calls this function with its worker rank.

    Parameters
    ----------
    dir_path: Output directory for Parquet files.
    env: CylonEnv object for this DataFrame
    name_function: a function to create the filename for that worker
    write_metadata_file: whether to write the metadata to _metadata file
    kwargs: the parameters that will be passed on to cudf.DataFrame.to_parquet function

    Returns
    -------
    Filename written
    """
    outfile = _determine_file_name_to_write(dir_path=dir_path, name_function=name_function, file_ext="parquet", env=env)
    df.to_cudf().to_parquet(outfile, **kwargs)

    if not write_metadata_file:
        return outfile

    fmd = pq.read_metadata(outfile)
    fmd.set_file_path(outfile)

    gather_root = 0
    buf = _serialize_metadata_file(fmd=fmd)
    buffers = gather_buffer(buf, root=gather_root, context=env.context)
    if gather_root != env.rank:
        return outfile

    fmd_all = _deserialize_metadata_buffers(buffers)
    fmd_single = fmd_all[0]
    [fmd_single.append_row_groups(fmd) for fmd in fmd_all[1:]]

    outfile_rp = os.path.realpath(outfile)
    meta_file = os.path.join(os.path.dirname(outfile_rp), "_metadata")
    fmd_single.write_metadata_file(meta_file)

    return outfile





