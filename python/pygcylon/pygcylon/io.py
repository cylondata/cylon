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
from pycylon.net.comm_ops import allgather_buffer


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


def _get_file_name_to_write(file_ext, file_names, env, **kwargs):
    """
    Write CSV or JSON file
    """
    if isinstance(file_names, str):
        file_ext = "_" + str(env.rank) + "." + file_ext
        out_file = os.path.join(file_names, "part" + file_ext) \
            if os.path.isdir(file_names) \
            else file_names + file_ext
        return out_file

    if isinstance(file_names, list) and \
            len(file_names) >= env.world_size and \
            all(isinstance(fn, str) for fn in file_names):
        return file_names[env.rank]

    if isinstance(file_names, dict) and \
            all(isinstance(key, int) and isinstance(val, str) for key, val in file_names.items()):
        return file_names[env.rank]

    raise ValueError("file_names must be: Union[str, List[str], Dict[int, str]]. "
                     + "When a list of strings provided, there must be at least one file name for each worker.")


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

    If a dictionary is provided as a paths parameter,
    workers with given ranks read those matching files and concatenate them.
    If a string or list of strings are provided, first the list of all files to read
    are determined by evaluating glob paths.
    Then files are sorted alphabetically and divided among the workers evenly.

    Parameters
    ----------
    paths: Input CSV file paths. A string, a list of strings,
           a dictionary of worker ranks to paths that can be either a string or a list of strings.
           paths are glob strings such as: path/to/dir/*.csv
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to cudf.read_csv function

    Returns
    -------
    A new distributed DataFrame constructed by reading all CSV files
    """
    return _read_csv_or_json(cudf.read_csv, paths=paths, env=env, **kwargs)


def write_csv(df: gcy.DataFrame,
              file_names: Union[str, List[str], Dict[int, str]],
              env: CylonEnv,
              **kwargs) -> str:
    """
    Write DataFrames to CSV files

    If a single string is provided as file_names:
    it can be either a file_base or a directory name.
      If it is a file base such as "path/to/dir/myfile",
        all workers add the extension "_<rank>.csv" to the file base.
      If the file_names is a directory:
        each worker create the output file by appending: "part_<rank>.csv"

    If a list of strings are provided: each string must be a filename for a worker.
      First string must be the output filename for the first worker,
      Second string must be the output filename for the second worker,
      etc.
      There must be one file name for each worker

    If a dictionary is provided:
      key must be the worker rank and the value must be the output file name for that worker.
      In this case, not all workers need to provide the output filenames for all worker
      If each worker provides its output filename only, that would be sufficient

    Parameters
    ----------
    df: DataFrame to write to files
    file_names: Output CSV file names.
                A string, or a list of strings, or a dictionary with worker ranks and out files
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to cudf.DataFrame.to_csv function

    Returns
    -------
    Filename written
    """
    outfile = _get_file_name_to_write(file_ext="csv", file_names=file_names, env=env, **kwargs)
    df.to_cudf().to_csv(outfile, **kwargs)
    return outfile


def read_json(paths: Union[str, List[str], Dict[int, Union[str, List[str]]]],
              env: CylonEnv,
              **kwargs) -> gcy.DataFrame:
    """
    Read JSON files and construct a distributed DataFrame

    If a dictionary is provided as a paths parameter,
    workers with given ranks read those matching files and concatenate them.
    If a string or list of strings are provided, first the list of all files to read
    are determined by evaluating glob paths.
    Then files are sorted alphabetically and divided among the workers evenly.

    Parameters
    ----------
    paths: Input JSON file paths. A string, a list of strings,
           a dictionary of worker ranks to paths that can be either a string or a list of strings.
           paths are glob strings such as: path/to/dir/*.json
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to cudf.read_json function

    Returns
    -------
    A new distributed DataFrame constructed by reading all JSON files
    """
    return _read_csv_or_json(cudf.read_json, paths=paths, env=env, **kwargs)


def write_json(df: gcy.DataFrame,
               file_names: Union[str, List[str], Dict[int, str]],
               env: CylonEnv,
               **kwargs) -> str:
    """
    Write DataFrames to JSON files

    If a single string is provided as file_names:
    it can be either a file_base or a directory name.
      If it is a file base such as "path/to/dir/myfile",
        all workers add the extension "_<rank>.json" to the file base.
      If the file_names is a directory:
        each worker create the output file by appending: "part_<rank>.json"

    If a list of strings are provided: each string must be a filename for a worker.
      First string must be the output filename for the first worker,
      Second string must be the output filename for the second worker,
      etc.
      There must be one file name for each worker

    If a dictionary is provided:
      key must be the worker rank and the value must be the output file name for that worker.
      In this case, not all workers need to provide the output filenames for all worker
      If each worker provides its output filename only, that would be sufficient

    Parameters
    ----------
    df: DataFrame to write to files
    file_names: Output JSON file names.
                A string, or a list of strings, or a dictionary with worker ranks and out files
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to cudf.DataFrame.to_json function

    Returns
    -------
    Filename written
    """
    outfile = _get_file_name_to_write(file_ext="json", file_names=file_names, env=env, **kwargs)
    df.to_cudf().to_json(outfile, **kwargs)
    return outfile


#####################################################################################
# Parquet io functions
#####################################################################################


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
    rows_per_worker = total_rows / env.world_size
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
    return df[end_index:] if last_worker else df[start_index:end_index]


def read_parquet(paths: Union[str, List[str], Dict[int, Union[str, List[str]]]],
                 env: CylonEnv,
                 **kwargs) -> gcy.DataFrame:
    """
    Read Parquet files and construct a distributed DataFrame
    Try to distribute rows evenly among the workers as much as possible
    """

    if isinstance(paths, str):
        paths = [paths]

    if isinstance(paths, list) and all(isinstance(p, str) for p in paths):
        worker_files = _get_worker_files(paths, env, ext="*.parquet")
        file_mappings_given = False
    elif isinstance(paths, dict) and \
            all(isinstance(key, int) and (isinstance(val, list) or isinstance(val, str)) for key, val in paths.items()):
        worker_files = _get_files(paths[env.rank], ext="*.parquet") if env.rank in paths else []
        file_mappings_given = True
    else:
        raise ValueError("paths must be: Union[str, List[str], Dict[int, Union[str, List[str]]]]")

    md_list = [pq.read_metadata(worker_file) for worker_file in worker_files]
    # make sure all schemas are equal for this worker
    for md1, md2, f1, f2 in zip(md_list, md_list[1:], worker_files, worker_files[1:]):
        if not md1.schema.equals(md2.schema):
            raise ValueError(env.rank, f"schemas are not equal for the files: {f1} and {f2}")

    # make sure all worker schemas are equal
    schema = md_list[0].schema.to_arrow_schema() if md_list else None
    df_schema = _all_schemas_equal(schema, env=env)

    if file_mappings_given:
        cdf = cudf.read_parquet(worker_files, **kwargs) if worker_files else \
            cudf.DataFrame.from_arrow(df_schema.empty_table())
        return gcy.DataFrame.from_cudf(cdf)

    # determine all local row_groups and put in a record_batch
    row_group_indices = []
    row_group_sizes = []
    files = []
    for md, worker_file in zip(md_list, worker_files):
        for rg in range(md.num_row_groups):
            row_group_indices.append(rg)
            row_group_sizes.append(md.row_group(rg).num_rows)
            files.append(worker_file)

    rg_data = [
               pa.array(row_group_indices, type="int64"),
               pa.array(row_group_sizes, type="int64"),
               pa.array(files, type="str")
               ]
    rg_batch = pa.record_batch(rg_data, names=['row_group_index', 'row_group_size', 'file'])

    # allgather row_groups
    # serialize the batch
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, rg_batch.schema) as writer:
        writer.write_batch(rg_batch)
    buf = sink.getvalue()

    buffers = allgather_buffer(buf=buf, context=env.context)

    # construct a single DataFrame from received batches
    rg_batches = []
    for rg_buf in buffers:
        if len(rg_buf) > 0:
            with pa.ipc.open_stream(rg_buf) as reader:
                rg_batches.append(reader.read_next_batch())

    if len(rg_batches) == 0:
        raise ValueError("No worker has any parquet files.")

    rg_table = pa.Table.from_batches(rg_batches)
    pdf = rg_table.to_pandas()

    # distribute row_groups to workers
    my_rg_df = _row_groups_this_worker(pdf, env)
    outfile = "row_groups_" + str(env.rank) + ".csv"
    my_rg_df.to_csv(outfile)
    print(env.rank, "written to the file: ", outfile)

    my_files = my_rg_df["file"].unique().tolist()
    my_row_groups = []
    for f in my_files:
        rg_per_file = my_rg_df[my_rg_df['file'] == f]["row_group_index"].tolist()
        my_row_groups.append(rg_per_file)

    if my_files:
        cdf = cudf.read_parquet(my_files, row_groups=my_row_groups, **kwargs)
    else:
        cdf = cudf.DataFrame.from_arrow(df_schema.empty_table())

    return gcy.DataFrame.from_cudf(cdf)


def write_parquet(df: gcy.DataFrame,
                  file_names: Union[str, List[str], Dict[int, str]],
                  env: CylonEnv,
                  **kwargs) -> str:
    """
    Write DataFrames to Parquet files

    If a single string is provided as file_names:
    it can be either a file_base or a directory name.
      If it is a file base such as "path/to/dir/myfile",
        all workers add the extension "_<rank>.parquet" to the file base.
      If the file_names is a directory:
        each worker create the output file by appending: "part_<rank>.parquet"

    If a list of strings are provided: each string must be a filename for a worker.
      First string must be the output filename for the first worker,
      Second string must be the output filename for the second worker,
      etc.
      There must be one file name for each worker

    If a dictionary is provided:
      key must be the worker rank and the value must be the output file name for that worker.
      In this case, not all workers need to provide the output filenames for all worker
      If each worker provides its output filename only, that would be sufficient

    Parameters
    ----------
    df: DataFrame to write to files
    file_names: Output Parquet file names.
                A string, or a list of strings, or a dictionary with worker ranks and out files
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to cudf.DataFrame.to_parquet function

    Returns
    -------
    Filename written
    """
    outfile = _get_file_name_to_write(file_ext="parquet", file_names=file_names, env=env, **kwargs)
    df.to_cudf().to_parquet(outfile, **kwargs)
    return outfile





