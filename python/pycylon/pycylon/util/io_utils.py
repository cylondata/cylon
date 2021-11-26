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

import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from pycylon.frame import CylonEnv
from pycylon.net.comm_ops import allgather_buffer, gather_buffer


def get_files(paths: Union[str, List[str]], ext: str = None) -> List[str]:
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


def indices_per_worker(nfiles, my_rank, nworkers):
    """
    calculate indices for a worker by evenly dividing files among them
    """
    files_per_worker = int(nfiles / nworkers)
    workers_with_exra_file = nfiles % nworkers
    start = files_per_worker * my_rank + \
            (my_rank if my_rank < workers_with_exra_file else workers_with_exra_file)
    end = start + files_per_worker + (1 if my_rank < workers_with_exra_file else 0)
    return start, end


def get_worker_files(paths: List[str], env: CylonEnv, ext: str = None) -> List[str]:
    """
    calculate the list of files that this worker will read
    """
    all_files = get_files(paths=paths, ext=ext)
    start, end = indices_per_worker(len(all_files), env.rank, env.world_size)
    return all_files[start:end]


def all_schemas_equal(schema: pyarrow.Schema, env: CylonEnv) -> pyarrow.Schema:
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


def gen_file_name(rank: int, file_ext: str) -> str:
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


def determine_file_name_to_write(dir_path, name_function, file_ext, env):
    """
    Determine the filename to write for a worker
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_name = gen_file_name(env.rank, file_ext) if name_function is None else name_function(env.rank)
    return os.path.join(dir_path, file_name)


#####################################################################################
# Parquet io functions
#####################################################################################


def serialize_metadata_file(fmd: pa._parquet.FileMetaData) -> pa.Buffer:
    """
    serialize a FileMetaData object
    return empty buffer if None
    """
    if fmd is None:
        return pa.allocate_buffer(0)
    else:
        _, buf_tuple = fmd.__reduce__()
        return buf_tuple[0]


def deserialize_metadata_buffers(buffers: List[pa.Buffer]) -> pa._parquet.FileMetaData:
    """
    deserialize FileMetaData buffers
    """
    fmd_all = []
    for received_buf in buffers:
        if received_buf.size > 0:
            fmd_all.append(pa._parquet._reconstruct_filemetadata(received_buf))
    return fmd_all


def schemas_equal(fmd_list: List[pa._parquet.FileMetaData], my_rank: int):
    """
    compare schemas of consecutive FileMetaData objects
    """
    for fmd1, fmd2 in zip(fmd_list, fmd_list[1:]):
        if not fmd1.schema.equals(fmd2.schema):
            f1 = fmd1.row_group(0).column(0).file_path
            f2 = fmd2.row_group(0).column(0).file_path
            raise ValueError(my_rank, f"schemas are not equal for the files: {f1} and {f2}")


def all_gather_metadata(worker_files: List[str], env: CylonEnv) -> pa._parquet.FileMetaData:
    """
    allgather FileMetaData objects and check schemas for equality
    """
    fmd_list = []
    for worker_file in worker_files:
        fmd = pq.read_metadata(worker_file)
        fmd.set_file_path(worker_file)
        fmd_list.append(fmd)

    # make sure all schemas are equal for this worker
    schemas_equal(fmd_list, env.rank)

    fmd_single = fmd_list[0] if fmd_list else None
    for fmd in fmd_list[1:]:
        fmd_single.append_row_groups(fmd)

    # serialize and all gather
    buf = serialize_metadata_file(fmd_single)
    buffers = allgather_buffer(buf=buf, context=env.context)

    # deserialize
    fmd_all = deserialize_metadata_buffers(buffers)
    if len(fmd_all) == 0:
        raise ValueError("No worker has any parquet files.")

    # make sure all received schemas are equal
    schemas_equal(fmd_all, env.rank)

    # combine all received schemas into one
    fmd_single = fmd_all[0]
    [fmd_single.append_row_groups(fmd) for fmd in fmd_all[1:]]
    return fmd_single


def row_groups_this_worker(df: pd.DataFrame, env: CylonEnv):
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


def construct_df(fmd: pa._parquet.FileMetaData) -> pd.DataFrame:
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


def gather_metadata_save(outfile, env: CylonEnv):
    fmd = pq.read_metadata(outfile)
    fmd.set_file_path(outfile)

    gather_root = 0
    buf = serialize_metadata_file(fmd=fmd)
    buffers = gather_buffer(buf, root=gather_root, context=env.context)
    if gather_root != env.rank:
        return outfile

    fmd_all = deserialize_metadata_buffers(buffers)
    fmd_single = fmd_all[0]
    [fmd_single.append_row_groups(fmd) for fmd in fmd_all[1:]]

    outfile_rp = os.path.realpath(outfile)
    meta_file = os.path.join(os.path.dirname(outfile_rp), "_metadata")
    fmd_single.write_metadata_file(meta_file)

