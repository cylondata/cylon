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

import cudf
import numpy as np
import pygcylon as gcy
from pycylon.frame import CylonEnv

from pygcylon.net.shuffle import shuffle as cshuffle
from pygcylon.net.c_comms import gather as cgather
from pygcylon.net.c_comms import allgather as callgather
from pygcylon.net.c_comms import broadcast as cbroadcast
from pygcylon.net.c_comms import repartition as crepartition


def shuffle(df: gcy.DataFrame, env: CylonEnv, on=None, ignore_index=False, index_shuffle=False) -> gcy.DataFrame:
    """
    Shuffle the distributed DataFrame by partitioning 'on' columns or index columns
    If this method is called with a single partition DataFrame, a copy of it is returned.

    Parameters
    ----------
    df: DataFrame to shuffle
    env: CylonEnv object for this DataFrame
    on: shuffling column name or names as a list
    ignore_index: ignore index when shuffling if True
    index_shuffle: shuffle on index columns if True (on and ingore_index parameters ignored if True)

    Returns
    -------
    A new distributed DataFrame constructed by shuffling the DataFrame
    """
    if env.world_size == 1:
        return gcy.DataFrame.from_cudf(cudf.DataFrame(df.to_cudf()))

    shuffle_column_indices = []

    # shuffle on index columns
    if index_shuffle:
        shuffle_column_indices = [*range(df.to_cudf()._num_indices)]
        ignore_index = False
    else:
        # make sure 'on' columns exist among data columns
        if (
                not np.iterable(on)
                or isinstance(on, str)
                or isinstance(on, tuple)
                and on in df.to_cudf()._data.names
        ):
            on = (on,)
        diff = set(on) - set(df.to_cudf()._data)
        if len(diff) != 0:
            raise ValueError(f"columns {diff} do not exist")

        # get indices of 'on' columns
        index_columns = 0 if ignore_index else df.to_cudf()._num_indices
        for name in on:
            shuffle_column_indices.append(index_columns + df.to_cudf()._column_names.index(name))

    tbl = cshuffle(df.to_cudf(), hash_columns=shuffle_column_indices, ignore_index=ignore_index, context=env.context)
    shuffled_cdf = cudf.DataFrame._from_table(tbl)
    return gcy.DataFrame.from_cudf(shuffled_cdf)


def gather(df: gcy.DataFrame,
           env: CylonEnv,
           gather_root: int = 0,
           ignore_index: bool = False,
           ) -> gcy.DataFrame:
    """
    Gather all dataframe partitions to a worker by keeping the global order of rows.
    For example:
        if there are 4 partitions currently with row counts: [10, 20 ,30 ,40]
        After gathering all partitions to the first worker,
        a new distributed dataframe is constructed with row counts: [100, 0 ,0 ,0]

    Parameters
    ----------
    df: DataFrame to gather
    env: CylonEnv object for this DataFrame
    gather_root: the worker rank to which all partitions will be gathered.
    ignore_index: ignore index when gathering if True

    Returns
    -------
    A new distributed DataFrame constructed by gathering all distributed dataframes to a single worker
    """
    if not isinstance(gather_root, int):
        raise ValueError("gather_root must be an int")

    if gather_root < 0 or gather_root >= env.world_size:
        raise ValueError(f"gather_root must be between [0, {env.world_size}]")

    if env.world_size == 1:
        return gcy.DataFrame.from_cudf(cudf.DataFrame(df.to_cudf()))

    gathered_tbl = cgather(df.to_cudf(),
                           context=env.context,
                           gather_root=gather_root,
                           ignore_index=ignore_index)
    # noinspection PyProtectedMember
    gathered_cdf = cudf.DataFrame._from_table(gathered_tbl)
    return gcy.DataFrame.from_cudf(gathered_cdf)


def allgather(df: gcy.DataFrame,
              env: CylonEnv,
              ignore_index: bool = False,
              ) -> gcy.DataFrame:
    """
    AllGather all dataframe partitions to all workers by keeping the global order of rows.
    For example:
        if there are 4 partitions currently with row counts: [10, 20 ,30 ,40]
        After allgathering all partitions to all workers,
        a new distributed dataframe is constructed with row counts: [100, 100 ,100 ,100]

    Parameters
    ----------
    df: DataFrame to gather
    env: CylonEnv object for this DataFrame
    ignore_index: ignore index when allgathering if True

    Returns
    -------
    A new distributed DataFrame constructed by allgathering all distributed dataframes to all workers
    """
    if env.world_size == 1:
        return gcy.DataFrame.from_cudf(cudf.DataFrame(df.to_cudf()))

    gathered_tbl = callgather(df.to_cudf(),
                              context=env.context,
                              ignore_index=ignore_index)
    # noinspection PyProtectedMember
    gathered_cdf = cudf.DataFrame._from_table(gathered_tbl)
    return gcy.DataFrame.from_cudf(gathered_cdf)


def broadcast(df: gcy.DataFrame,
              env: CylonEnv,
              root: int,
              ignore_index: bool = False,
              ) -> gcy.DataFrame:
    """
    Broadcast a dataframe partition to all workers.
    For example:
        Assuming there are 4 partitions currently with row counts: [10, 20 ,30 ,40]
        If we broadcast the partition of the second worker,
        a new distributed dataframe is constructed with identical DataFrames in all workers
        with row counts: [20, 20 ,20 ,20]

    Parameters
    ----------
    df: DataFrame to gather
    env: CylonEnv object for this DataFrame
    root: the worker rank from which the DataFrame will be send out to all others.
    ignore_index: ignore index when broadcasting if True

    Returns
    -------
    A new distributed DataFrame constructed by broadcasting a dataframe to all workers
    """
    if not isinstance(root, int):
        raise ValueError("broadcast root must be an int")

    if root < 0 or root >= env.world_size:
        raise ValueError(f"root must be between [0, {env.world_size}]")

    if env.world_size == 1:
        return gcy.DataFrame.from_cudf(cudf.DataFrame(df.to_cudf()))

    bcast_tbl = cbroadcast(df.to_cudf(),
                           context=env.context,
                           root=root,
                           ignore_index=ignore_index)
    # noinspection PyProtectedMember
    bcast_cdf = cudf.DataFrame._from_table(bcast_tbl)
    return gcy.DataFrame.from_cudf(bcast_cdf)


def repartition(df: gcy.DataFrame,
                env: CylonEnv,
                rows_per_partition: List[int] = None,
                ignore_index: bool = False,
                ) -> gcy.DataFrame:
    """
    Repartition the dataframe by keeping the global order of rows.
    If rows_per_partition is not provided, repartition the rows evenly among the workers
    The sum of rows in rows_per_partition must match
    the total number of rows in the current distributed dataframe.
    For example:
        if there are 4 partitions currently with row counts: [10, 20 ,30 ,40]
        After repartitioning evenly, row counts become: [25, 25 ,25 ,25]

        if the number of partitions is not a multiple of sum of row counts,
          first k partitions get one more row:
          if there are 4 partitions currently with row counts: [10, 20 ,30 ,42]
          After repartitioning evenly, row counts become: [26, 26 ,25 ,25]

    User Provided Row Counts:
        if there are 4 partitions currently with row counts: [10, 20 ,30 ,40]
        Users can request repartitioning with the row counts of [35, 15 ,20 ,30],
        the repartitioned dataframe will have the requested row counts.

        Requested row count list must have integer values for each partition and
        the total number of rows must match the current row count in the distributed source dataframe

    Parameters
    ----------
    df: the DataFrame to repartition
    env: CylonEnv object for this DataFrame
    rows_per_partition: list of partition sizes requested after the repartitioning
    ignore_index: ignore index when repartitioning if True

    Returns
    -------
    A new distributed DataFrame constructed by repartitioning the DataFrame
    """
    # make sure 'rows_per_partition' consists of ints and its size matches number of workers
    if rows_per_partition is not None:
        if not all(isinstance(i, int) for i in rows_per_partition):
            raise ValueError("rows_per_partition is not a list of int")

        if len(rows_per_partition) != env.world_size:
            raise ValueError("len(rows_per_partition) must match the number of workers")

    if env.world_size == 1:
        if rows_per_partition is None or len(df.to_cudf()) == rows_per_partition[0]:
            return gcy.DataFrame.from_cudf(cudf.DataFrame(df.to_cudf()))
        else:
            raise ValueError(f"row counts do not match. current dataframe has {len(df.to_cudf())}, "
                             + f"requested row count: {rows_per_partition[0]}")

    reparted_tbl = crepartition(df.to_cudf(),
                                context=env.context,
                                rows_per_worker=rows_per_partition,
                                ignore_index=ignore_index)
    # noinspection PyProtectedMember
    reparted_cdf = cudf.DataFrame._from_table(reparted_tbl)
    return gcy.DataFrame.from_cudf(reparted_cdf)


