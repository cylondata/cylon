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
import pygcylon as gcy
from pycylon.frame import CylonEnv


def _get_files(paths: Union[str, List[str]]) -> List[str]:
    """
    convert a list of glob strings to a list of files
    get all file names as a single list
    sort them alphabetically and return
    """
    if isinstance(paths, str):
        return sorted(glob.glob(paths))

    all_files = []
    for p in paths:
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


def _get_worker_files(paths: List[str], env: CylonEnv) -> List[str]:
    """
    calculate the list of files that this worker will read
    """
    all_files = _get_files(paths=paths)
    start, end = _indices_per_worker(len(all_files), env.rank, env.world_size)
    return all_files[start:end]


def _get_first_file(paths: Union[List[str], Dict[int, Union[str, List[str]]]]) -> str:
    """
    get the first file in the given paths if exist
    """
    all_files = []
    if isinstance(paths, list):
        all_files = _get_files(paths=paths)

    if isinstance(paths, dict):
        all_paths = []
        for val in paths.values():
            all_paths.extend(val) if isinstance(val, list) else all_paths.append(val)
        all_files = _get_files(paths=all_paths)

    return all_files[0] if all_files else None


def _all_schemas_equal(df: cudf.DataFrame, env: CylonEnv) -> bool:
    """
    after reading files, check whether all DataFrame schemas are equal
    todo: cudf is adding serialize/deserialize methods to DataFrame in cudf 21.12
          maybe we can do this after when we switch to cudf 21.12
    """
    pass


def _write_csv_or_json(write_fun, file_ext, file_names, env, **kwargs):
    """
    Write CSV or JSON file
    this is to avoid repetitive code
    """
    if isinstance(file_names, str):
        out_file = file_names + "/part_" + str(env.rank) + "." + file_ext \
            if os.path.isdir(file_names) \
            else file_names + "_" + str(env.rank) + "." + file_ext
        write_fun(out_file, **kwargs)
        return out_file

    if isinstance(file_names, list) and \
            len(file_names) >= env.world_size and \
            all(isinstance(fn, str) for fn in file_names):
        write_fun(file_names[env.rank], **kwargs)
        return file_names[env.rank]
    else:
        raise ValueError("file_names must be: Union[str, List[str]]. "
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

    if not worker_files:
        first_file = _get_first_file(paths=paths)
        if not first_file:
            raise ValueError("No files to read.")
        # does not work for json if lines=True is not set
        cdf = read_fun(first_file, nrows=0, **kwargs)
        return gcy.DataFrame.from_cudf(cdf)

    # if there is only one file to read
    if len(worker_files) == 1:
        cdf = read_fun(worker_files[0], **kwargs)
        return gcy.DataFrame.from_cudf(cdf)

    cdf_list = []
    for worker_file in worker_files:
        cdf_list.append(read_fun(worker_file, **kwargs))

    cdf = cudf.concat(cdf_list)
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
              file_names: Union[str, List[str]],
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

    Parameters
    ----------
    df: DataFrame to write to files
    file_names: Output CSV file names. A string, or a list of strings,
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to cudf.write_csv function

    Returns
    -------
    Filename written
    """
    return _write_csv_or_json(df.to_cudf().to_csv, file_ext="csv", file_names=file_names, env=env, **kwargs)


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
               file_names: Union[str, List[str]],
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

    Parameters
    ----------
    df: DataFrame to write to files
    file_names: Output JSON file names. A string, or a list of strings,
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to cudf.write_json function

    Returns
    -------
    Filename written
    """
    return _write_csv_or_json(df.to_cudf().to_json, file_ext="json", file_names=file_names, env=env, **kwargs)








