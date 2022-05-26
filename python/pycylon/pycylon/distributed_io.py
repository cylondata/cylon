from typing import List, Dict, Union
import os

from pycylon.frame import CylonEnv, DataFrame, concat, read_csv
from pycylon.util import io_utils
import pyarrow.parquet as pq
import pyarrow.csv as pac
import pandas as pd

def _read_csv_or_json(read_fun, paths, env, **kwargs) -> DataFrame:
    """
    Read CSV or JSON files and construct a distributed DataFrame
    """

    if isinstance(paths, str):
        paths = [paths]

    if isinstance(paths, list) and all(isinstance(p, str) for p in paths):
        worker_files = io_utils.get_worker_files(paths, env)
    elif isinstance(paths, dict) and \
            all(isinstance(key, int) and (isinstance(val, list) or isinstance(val, str)) for key, val in paths.items()):
        worker_files = io_utils.get_files(paths[env.rank]) if env.rank in paths else []
    else:
        raise ValueError("paths must be: Union[str, List[str], Dict[int, Union[str, List[str]]]]")

    df = None
    # if there is only one file to read
    if len(worker_files) == 1:
        df = read_fun(worker_files[0], **kwargs)
    elif len(worker_files) > 1:
        df_list = []
        for worker_file in worker_files:
            df_list.append(read_fun(worker_file, **kwargs))
        df = concat(df_list)

    schema = df.to_arrow().schema if df is not None else None
    df_schema = io_utils.all_schemas_equal(schema, env=env)
    if df is None:
        df = DataFrame(df_schema.empty_table())

    return df


def read_csv_dist(paths: Union[str, List[str], Dict[int, Union[str, List[str]]]],
            env: CylonEnv, **kwargs) -> DataFrame:
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
        pycylon.read_csv_dist(paths="/pat/to/dir/*.csv", env=env)

        # read only a single csv file
        pycylon.read_csv_dist(paths="/pat/to/dir/example.csv", env=env)

        # read only a list of csv files
        pycylon.read_csv_dist(paths=["/pat/to/dir/example_0.csv", "/pat/to/dir/example_1.csv"], env=env)

        # specify the files to be read by individual workers
        pycylon.read_csv_dist(paths={0: "/pat/to/dir/example_0.csv", 1: "/pat/to/dir/example_1.csv"}, env=env)
        pycylon.read_csv_dist(paths={0: "/pat/to/dir/example_0*.csv", 1: "/pat/to/dir/example_1*.csv"}, env=env)

        # specify many files to be read by individual workers
        pycylon.read_csv_dist(paths={0: ["/pat/to/dir/example_0.csv", "/pat/to/dir/example_1*.csv"],
                            1: "/pat/to/dir/example_2*.csv"}, env=env)

    Parameters
    ----------
    paths: Input CSV file paths as glob strings.
           A string, a list of strings, a dictionary of worker ranks to paths
           that can be either a string or a list of strings.
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to pycylon.frame.read_csv function

    Returns
    -------
    A new distributed DataFrame constructed by reading all CSV files
    """
    return _read_csv_or_json(read_csv, paths, env, **kwargs)

def read_json_dist(paths: Union[str, List[str], Dict[int, Union[str, List[str]]]],
            env: CylonEnv, **kwargs) -> DataFrame:
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
        pycylon.read_json_dist(paths="/pat/to/dir/*.json", env=env)

        # read only a single json file
        pycylon.read_json_dist(paths="/pat/to/dir/example.json", env=env)

        # read only a list of json files
        pycylon.read_json_dist(paths=["/pat/to/dir/example_0.json", "/pat/to/dir/example_1.json"], env=env)

        # specify the files to be read by individual workers
        pycylon.read_json_dist(paths={0: "/pat/to/dir/example_0.json", 1: "/pat/to/dir/example_1.json"}, env=env)
        pycylon.read_json_dist(paths={0: "/pat/to/dir/example_0*.json", 1: "/pat/to/dir/example_1*.json"}, env=env)

        # specify many files to be read by individual workers
        pycylon.read_json_dist(paths={0: ["/pat/to/dir/example_0.json", "/pat/to/dir/example_1*.json"],
                            1: "/pat/to/dir/example_2*.json"}, env=env)

    Parameters
    ----------
    paths: Input JSON file paths as glob strings.
           A string, a list of strings, a dictionary of worker ranks to paths
           that can be either a string or a list of strings.
    env: CylonEnv object for this DataFrame
    kwargs: the parameters that will be passed on to pandas.read_json function

    Returns
    -------
    A new distributed DataFrame constructed by reading all JSON files
    """
    def json_read_fun(path, **kwargs):
        return DataFrame(pd.read_json(path, **kwargs))
    return _read_csv_or_json(json_read_fun, paths, env, **kwargs)

def read_parquet_dist(paths: Union[str, List[str], Dict[int, Union[str, List[str]]]], env: CylonEnv,
                metadata=None, common_metadata=None, read_dictionary=None,
                memory_map=False, buffer_size=0, pre_buffer=False,
                coerce_int96_timestamp_unit=None,
                columns=None, use_threads=False, use_pandas_metadata=False, **kwargs) -> DataFrame:
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
        pycylon.read_parquet_dist(paths="/pat/to/dir/", env=env)

        # read all parquet files in a directory
        pycylon.read_parquet_dist(paths="/pat/to/dir/*.parquet", env=env)

        # read only a single parquet file
        pycylon.read_parquet_dist(paths="/pat/to/dir/example.parquet", env=env)

        # read only a list of parquet files
        pycylon.read_parquet_dist(paths=["/pat/to/dir/example_0.parquet", "/pat/to/dir/example_1.parquet"], env=env)

        # specify the files to be read by individual workers
        pycylon.read_parquet_dist(paths={0: "/pat/to/dir/example_0.parquet", 1: "/pat/to/dir/example_1.parquet"}, env=env)
        pycylon.read_parquet_dist(paths={0: "/pat/to/dir/example_0*.parquet", 1: "/pat/to/dir/example_1*.parquet"}, env=env)

        # specify many files to be read by individual workers
        pycylon.read_parquet_dist(paths={0: ["/pat/to/dir/example_0.parquet", "/pat/to/dir/example_1*.parquet"],
                            1: "/pat/to/dir/example_2*.parquet"}, env=env)

    Parameters
    ----------
    paths: Input parquet file paths as glob strings.
           A string, a list of strings, a dictionary of worker ranks to paths
           that can be either a string or a list of strings.
    env: CylonEnv object for this DataFrame
    keyword arguments: keyword args that are passed to pyarrow.ParquetFile and pyarrow.ParquetFile.read_row_groups
    kwargs: other keyword args for pq.read_table

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
            worker_files = io_utils.get_worker_files(paths, env, ext="*.parquet")
            file_mappings_given = False
        elif isinstance(paths, dict) and \
                all(isinstance(key, int) and (isinstance(val, list) or isinstance(val, str)) for key, val in paths.items()):
            worker_files = io_utils.get_files(paths[env.rank], ext="*.parquet") if env.rank in paths else []
            file_mappings_given = True
        else:
            raise ValueError("paths must be: Union[str, List[str], Dict[int, Union[str, List[str]]]]")

        fmd = io_utils.all_gather_metadata(worker_files=worker_files, env=env)
        if file_mappings_given:
            if not worker_files:
                return DataFrame(fmd.schema.to_arrow_schema().empty_table())
            dfs = []
            for file in worker_files:
                dfs.append(DataFrame(pq.read_table(file, 
                # keyword arguments to pass into pq.read_table
                metadata=metadata, read_dictionary=read_dictionary,
                memory_map=memory_map, buffer_size=buffer_size, pre_buffer=pre_buffer,
                coerce_int96_timestamp_unit=coerce_int96_timestamp_unit,
                columns=columns, use_threads=use_threads, use_pandas_metadata=use_pandas_metadata, **kwargs)))
            return concat(dfs)

    # construct a DataFrame with relevant columns
    pdf = io_utils.construct_df(fmd)

    # distribute row_groups to workers
    my_rg_df = io_utils.row_groups_this_worker(pdf, env)

    my_files = my_rg_df["file"].unique().tolist()

    if not my_files:
        return DataFrame(fmd.schema.to_arrow_schema().empty_table())

    dfs = []
    for f in my_files:
        pq_file = pq.ParquetFile(f,
            # keyword args to pass to pq.ParquetFilee
            metadata=metadata, common_metadata=common_metadata, read_dictionary=read_dictionary,
            memory_map=memory_map, buffer_size=buffer_size, pre_buffer=pre_buffer,
            coerce_int96_timestamp_unit=coerce_int96_timestamp_unit)

        rg = my_rg_df[my_rg_df['file'] == f]["row_group_index"].tolist()

        ptable = pq_file.read_row_groups(rg, 
        # keyword args to pass to pq.ParquetFile.read_row_groups
        columns=columns, use_threads=use_threads, use_pandas_metadata=use_pandas_metadata)
        dfs.append(DataFrame(ptable))

    return concat(dfs)

def write_csv_dist(df: DataFrame,
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
    kwargs: the parameters that will be passed on to pyarrow.csv.write_csv function

    Returns
    -------
    Filename written
    """
    outfile = io_utils.determine_file_name_to_write(dir_path=dir_path,
                                                    name_function=name_function,
                                                    file_ext="csv",
                                                    env=env)
    pac.write_csv(df.to_arrow(), outfile, **kwargs)
    return outfile

def write_json_dist(df: DataFrame,
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
    kwargs: the parameters that will be passed on to pandas.Dataframe.to_json function

    Returns
    -------
    Filename written
    """
    # convert to pandas dataframe b/c no to_json function available in pyarrow
    outfile = io_utils.determine_file_name_to_write(dir_path=dir_path, name_function=name_function, file_ext="json", env=env)
    pdf = df.to_pandas()
    pdf.to_json(outfile, **kwargs)
    return outfile

def write_parquet_dist(df: DataFrame,
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
    kwargs: the parameters that will be passed on to pyarrow.parquet.write_table function

    Returns
    -------
    Filename written
    """
    outfile = io_utils.determine_file_name_to_write(dir_path=dir_path,
                                                name_function=name_function,
                                                file_ext="parquet",
                                                env=env)
    pq.write_table(df.to_arrow(), outfile, **kwargs)

    if not write_metadata_file:
        return outfile

    io_utils.gather_metadata_save(outfile, env=env)
    return outfile