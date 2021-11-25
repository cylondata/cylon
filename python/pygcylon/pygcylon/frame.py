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
from pygcylon.net.shuffle import shuffle as cshuffle
from pygcylon.net.row_counts import row_counts_all_tables
from pygcylon.net.sorting import distributed_sort
from pycylon.frame import CylonEnv
from pygcylon.groupby import GroupByDataFrame
import pygcylon.comms as comms
import pygcylon.io


class DataFrame(object):

    def __init__(self, data=None, index=None, columns=None, dtype=None):
        """
        A GPU Dataframe object.

        Parameters
        ----------
        data : array-like, Iterable, dict, or DataFrame.
            Dict can contain Series, arrays, constants, or list-like objects.

        index : Index or array-like
            Index to use for resulting frame. Will default to
            RangeIndex if no indexing information part of input data and
            no index provided.

        columns : Index or array-like
            Column labels to use for resulting frame.
            Will default to RangeIndex (0, 1, 2, …, n) if no column
            labels are provided.

        dtype : dtype, default None
            Data type to force. Only a single dtype is allowed.
            If None, infer.
        """
        self._cdf = cudf.DataFrame(data=data, index=index, columns=columns, dtype=dtype)

    def __repr__(self):
        return self._cdf.__repr__()

    def __str__(self):
        return self._cdf.__str__()

    def __setitem__(self, key, value):
        self._cdf.__setitem__(arg=key, value=value)

    def __getitem__(self, arg):
        result = self._cdf.__getitem__(arg=arg)
        return DataFrame.from_cudf(result) if isinstance(result, cudf.DataFrame) else result

    def __setattr__(self, key, col):
        if key == "_cdf":
            super().__setattr__(key, col) if isinstance(col, cudf.DataFrame) \
                else ValueError("_cdf has to be an instance of cudf.DataFrame")
        elif self._cdf:
            self._cdf.__setattr__(key=key, col=col)
        else:
            raise ValueError("Invalid attribute setting attempt")

    def __getattr__(self, key):
        return self._cdf if key == "_cdf" else self._cdf.__getattr__(key=key)

    def __delitem__(self, name):
        self._cdf.__delitem__(name=name)

    def __dir__(self):
        return self._cdf.__dir__()

    def __sizeof__(self):
        return self._cdf.__sizeof__()

    def __len__(self):
        return self._cdf.__len__()

    def __del__(self):
        del self._cdf

    @staticmethod
    def from_cudf(cdf) -> DataFrame:
        if (cdf is not None) and isinstance(cdf, cudf.DataFrame):
            df = DataFrame()
            df._cdf = cdf
            return df
        else:
            raise ValueError('A cudf.DataFrame object must be provided.')

    def to_cudf(self) -> cudf.DataFrame:
        return self._cdf

    def join(self,
             other,
             on=None,
             how='left',
             lsuffix='l',
             rsuffix='r',
             sort=False,
             algorithm="hash",
             env: CylonEnv = None) -> DataFrame:
        """Join columns with other DataFrame on index column.

        Parameters
        ----------
        other : DataFrame
        how : str
            Only accepts "left", "right", "inner", "outer"
        lsuffix, rsuffix : str
            The suffices to add to the left (*lsuffix*) and right (*rsuffix*)
            column names when avoiding conflicts.
        sort : bool
            Set to True to ensure sorted ordering.

        Returns
        -------
        joined : DataFrame

        Notes
        -----
        Difference from pandas:

        - *other* must be a single DataFrame for now.
        - *on* is not supported yet due to lack of multi-index support.
        """

        if on is not None:
            raise ValueError('on is not supported with join method. Please use merge method.')

        if env is None:
            joined_df = self._cdf.join(other=other._cdf,
                                       on=on,
                                       how=how,
                                       lsuffix=lsuffix,
                                       rsuffix=rsuffix,
                                       sort=sort,
                                       method=algorithm)
            return DataFrame.from_cudf(joined_df)

        # shuffle dataframes on index columns
        hash_columns = [*range(self._cdf._num_indices)]
        shuffled_left = _shuffle(self._cdf, hash_columns=hash_columns, env=env)

        hash_columns = [*range(other._cdf._num_indices)]
        shuffled_right = _shuffle(other._cdf, hash_columns=hash_columns, env=env)

        joined_df = shuffled_left.join(shuffled_right,
                                       on=on,
                                       how=how,
                                       lsuffix=lsuffix,
                                       rsuffix=rsuffix,
                                       sort=sort,
                                       method=algorithm)
        return DataFrame.from_cudf(joined_df)

    def merge(self,
              right,
              how="inner",
              algorithm="hash",
              on=None,
              left_on=None,
              right_on=None,
              left_index=False,
              right_index=False,
              sort=False,
              suffixes=("_x", "_y"),
              copy=True,
              indicator=False,
              validate=None,
              env: CylonEnv = None) -> DataFrame:
        """Merge GPU DataFrame objects by performing a database-style join
        operation by columns or indexes.

        Parameters
        ----------
        right : DataFrame
        on : label or list; defaults to None
            Column or index level names to join on. These must be found in
            both DataFrames.

            If on is None and not merging on indexes then
            this defaults to the intersection of the columns
            in both DataFrames.
        how : {‘left’, ‘outer’, ‘inner’}, default ‘inner’
            Type of merge to be performed.

            - left : use only keys from left frame, similar to a SQL left
              outer join.
            - right : not supported.
            - outer : use union of keys from both frames, similar to a SQL
              full outer join.
            - inner: use intersection of keys from both frames, similar to
              a SQL inner join.
        left_on : label or list, or array-like
            Column or index level names to join on in the left DataFrame.
            Can also be an array or list of arrays of the length of the
            left DataFrame. These arrays are treated as if they are columns.
        right_on : label or list, or array-like
            Column or index level names to join on in the right DataFrame.
            Can also be an array or list of arrays of the length of the
            right DataFrame. These arrays are treated as if they are columns.
        left_index : bool, default False
            Use the index from the left DataFrame as the join key(s).
        right_index : bool, default False
            Use the index from the right DataFrame as the join key.
        sort : bool, default False
            Sort the resulting dataframe by the columns that were merged on,
            starting from the left.
        suffixes: Tuple[str, str], defaults to ('_x', '_y')
            Suffixes applied to overlapping column names on the left and right
            sides
        algorithm: {‘hash’, ‘sort’}, default ‘hash’
            The implementation method to be used for the operation.

        Returns
        -------
            merged : DataFrame

        Notes
        -----
        **DataFrames merges in cuDF result in non-deterministic row ordering.**

        Examples
        --------
        >>> import cudf
        >>> df_a = cudf.DataFrame()
        >>> df_a['key'] = [0, 1, 2, 3, 4]
        >>> df_a['vals_a'] = [float(i + 10) for i in range(5)]
        >>> df_b = cudf.DataFrame()
        >>> df_b['key'] = [1, 2, 4]
        >>> df_b['vals_b'] = [float(i+10) for i in range(3)]
        >>> cdf_a = DataFrame(df_a)
        >>> cdf_b = DataFrame(df_b)
        >>> cdf_merged = cdf_a.merge(cdf_b, on=['key'], how='left')
        >>> cdf_merged.sort_values('key')  # doctest: +SKIP
           key  vals_a  vals_b
        3    0    10.0
        0    1    11.0    10.0
        1    2    12.0    11.0
        4    3    13.0
        2    4    14.0    12.0

        **Merging on categorical variables is only allowed in certain cases**

        Categorical variable typecasting logic depends on both `how`
        and the specifics of the categorical variables to be merged.
        Merging categorical variables when only one side is ordered
        is ambiguous and not allowed. Merging when both categoricals
        are ordered is allowed, but only when the categories are
        exactly equal and have equal ordering, and will result in the
        common dtype.
        When both sides are unordered, the result categorical depends
        on the kind of join:
        - For inner joins, the result will be the intersection of the
        categories
        - For left or right joins, the result will be the the left or
        right dtype respectively. This extends to semi and anti joins.
        - For outer joins, the result will be the union of categories
        from both sides.
        """

        if indicator:
            raise NotImplemented(
                "Only indicator=False is currently supported"
            )

        if env is None or env.world_size == 1:
            merged_df = self._cdf.merge(right=right._cdf,
                                        on=on,
                                        left_on=left_on,
                                        right_on=right_on,
                                        left_index=left_index,
                                        right_index=right_index,
                                        how=how,
                                        sort=sort,
                                        suffixes=suffixes,
                                        method=algorithm)
            return DataFrame.from_cudf(merged_df)

        from cudf.core.join.join import Merge
        # just for checking purposes, we assign "left" to how if it is "right"
        howToCheck = "left" if how == "right" else how
        Merge._validate_merge_params(lhs=self._cdf,
                                     rhs=right._cdf,
                                     on=on,
                                     left_on=left_on,
                                     right_on=right_on,
                                     left_index=left_index,
                                     right_index=right_index,
                                     how=howToCheck,
                                     suffixes=suffixes)

        left_on1, right_on1 = self._get_left_right_on(self._cdf,
                                                      right._cdf,
                                                      on,
                                                      left_on,
                                                      right_on,
                                                      left_index,
                                                      right_index)
        left_on_ind, right_on_ind = DataFrame._get_left_right_indices(self._cdf,
                                                                      right._cdf,
                                                                      left_on1,
                                                                      right_on1,
                                                                      left_index,
                                                                      right_index)

        shuffled_left = _shuffle(self._cdf, hash_columns=left_on_ind, env=env)
        shuffled_right = _shuffle(right._cdf, hash_columns=right_on_ind, env=env)

        merged_df = shuffled_left.merge(right=shuffled_right,
                                        on=on,
                                        left_on=left_on,
                                        right_on=right_on,
                                        left_index=left_index,
                                        right_index=right_index,
                                        how=how,
                                        sort=sort,
                                        suffixes=suffixes,
                                        method=algorithm)
        return DataFrame.from_cudf(merged_df)

    @staticmethod
    def _get_left_right_on(lhs, rhs, on, left_on, right_on, left_index, right_index):
        """
        Calculate left_on and right_on as a list of strings (column names)
        this is based on the "preprocess_merge_params" function in the cudf file:
            cudf/core/join/join.py
        """
        if on:
            on = [on] if isinstance(on, str) else list(on)
            left_on = right_on = on
        else:
            if left_on:
                left_on = (
                    [left_on] if isinstance(left_on, str) else list(left_on)
                )
            if right_on:
                right_on = (
                    [right_on] if isinstance(right_on, str) else list(right_on)
                )

        same_named_columns = [value for value in lhs._column_names if value in rhs._column_names]
        if not (left_on or right_on) and not (left_index and right_index):
            left_on = right_on = list(same_named_columns)

        return left_on, right_on

    @staticmethod
    def _get_left_right_indices(lhs, rhs, left_on, right_on, left_index, right_index):
        """
        Calculate left and right column indices to perform shuffle on
        this is based on the "join" function in cudf file:
            cudf/_lib/join.pyx
        """

        if left_on is None:
            left_on = []
        if right_on is None:
            right_on = []

        left_on_ind = []
        right_on_ind = []

        if left_index or right_index:
            # If either true, we need to process both indices as columns
            left_join_cols = list(lhs._index_names) + list(lhs._column_names)
            right_join_cols = list(rhs._index_names) + list(rhs._column_names)

            if left_index and right_index:
                # Both dataframes must take index column indices
                left_on_indices = right_on_indices = range(lhs._num_indices)

            elif left_index:
                # Joins left index columns with right 'on' columns
                left_on_indices = range(lhs._num_indices)
                right_on_indices = [
                    right_join_cols.index(on_col) for on_col in right_on
                ]

            elif right_index:
                # Joins right index columns with left 'on' columns
                right_on_indices = range(rhs._num_indices)
                left_on_indices = [
                    left_join_cols.index(on_col) for on_col in left_on
                ]

            for i_l, i_r in zip(left_on_indices, right_on_indices):
                left_on_ind.append(i_l)
                right_on_ind.append(i_r)

        else:
            left_join_cols = list(lhs._index_names) + list(lhs._column_names)
            right_join_cols = list(rhs._index_names) + list(rhs._column_names)

        # If both left/right_index, joining on indices plus additional cols
        # If neither, joining on just cols, not indices
        # In both cases, must match up additional column indices in lhs/rhs
        if left_index == right_index:
            for name in left_on:
                left_on_ind.append(left_join_cols.index(name))
            for name in right_on:
                right_on_ind.append(right_join_cols.index(name))

        return left_on_ind, right_on_ind

    def _get_column_indices(self) -> List[int]:
        """
        Get the column indices excluding index columns
        :return: list of ints
        """
        lists = DataFrame._get_all_column_indices([self])
        return lists[0]

    @staticmethod
    def _get_all_column_indices(dfs) -> List[List[int]]:
        """
        Get indices of all DataFrames excluding index columns
        This is to calculate indices of columns that will be used
        to perform partitioning/shuffling on the dataframe
        :param dfs: list of DataFrame objects
        :return: list of list of column indices
        """
        all_df_indices = [];
        for cdf in dfs:
            df_indices = [*range(cdf._cdf._num_indices, cdf._cdf._num_indices + cdf._cdf._num_columns)]
            all_df_indices.append(df_indices)
        return all_df_indices

    @staticmethod
    def _get_all_common_indices(dfs) -> List[List[int]]:
        """
        Get indices of all columns common in all DataFrames
        Columns might be in different indices in different DataFrames
        This is to calculate indices of columns that will be used
        to perform partitioning/shuffling on the dataframe
        :param dfs: list of DataFrame objects
        :return: list of list of column indices
        """

        # get the inersection of all column names
        common_columns_names = DataFrame._get_common_column_names(dfs)
        if len(common_columns_names) == 0:
            raise ValueError("There is no common column names among the provided DataFrame objects")

        all_df_indices = [];
        for cdf in dfs:
            df_indices = []
            col_names = list(cdf._cdf._index_names) + list(cdf._cdf._column_names)
            for name in common_columns_names:
                df_indices.append(col_names.index(name))
            all_df_indices.append(df_indices)
        return all_df_indices

    @staticmethod
    def _get_common_column_names(dfs) -> List[str]:
        """
        Get common column names in the proved DataFrames
        :param dfs: list of DataFrame objects
        :return: list of column names that are common to all DataFrames
        """
        column_name_lists = [list(obj._cdf._column_names) for obj in dfs]
        common_column_names = set(column_name_lists[0])
        for column_names in column_name_lists[1:]:
            common_column_names = common_column_names & set(column_names)
        return common_column_names

    def drop_duplicates(
            self,
            subset: Optional[Union[Hashable, Sequence[Hashable]]] = None,
            keep: Union[str, bool] = "first",
            inplace: bool = False,
            ignore_index: bool = False,
            env: CylonEnv = None) -> Union[DataFrame or None]:
        """
        Remove duplicate rows from the DataFrame.
        Considering certain columns is optional. Indexes, including time indexes
        are ignored.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', False}, default 'first'
            Determines which duplicates (if any) to keep.
            - ``first`` : Drop duplicates except for the first occurrence.
            - ``last`` : Drop duplicates except for the last occurrence.
            - False: Drop all duplicates.
        inplace : bool, default False
            Whether to drop duplicates in place or to return a copy.
            inplace is supported only in local mode
            when there are multiple workers in the computation, inplace is disabled
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
        env: CylonEnv object

        Returns
        -------
        DataFrame or None
            DataFrame with duplicates removed or
            None if ``inplace=True`` and in the local mode with no distributed workers.
        """

        subset = self._convert_subset(subset=subset, ignore_len_check=True)

        if env is None or env.world_size == 1:
            dropped_df = self._cdf.drop_duplicates(subset=subset, keep=keep, inplace=inplace, ignore_index=ignore_index)
            return DataFrame.from_cudf(dropped_df) if not inplace else None

        shuffle_column_indices = []
        for name in subset:
            shuffle_column_indices.append(self._cdf._num_indices + self._cdf._column_names.index(name))

        shuffled_df = _shuffle(self._cdf, hash_columns=shuffle_column_indices, env=env)

        dropped_df = shuffled_df.drop_duplicates(subset=subset, keep=keep, inplace=inplace, ignore_index=ignore_index)
        return DataFrame.from_cudf(shuffled_df) if inplace else DataFrame.from_cudf(dropped_df)

    def set_index(
            self,
            keys,
            drop=True,
            append=False,
            inplace=False,
            verify_integrity=False,
    ) -> Union[DataFrame or None]:
        """Return a new DataFrame with a new index

        Parameters
        ----------
        keys : Index, Series-convertible, label-like, or list
            Index : the new index.
            Series-convertible : values for the new index.
            Label-like : Label of column to be used as index.
            List : List of items from above.
        drop : boolean, default True
            Whether to drop corresponding column for str index argument
        append : boolean, default True
            Whether to append columns to the existing index,
            resulting in a MultiIndex.
        inplace : boolean, default False
            Modify the DataFrame in place (do not create a new object).
        verify_integrity : boolean, default False
            Check for duplicates in the new index.

        Returns
        -------
        DataFrame or None
            DataFrame with a new index or
            None if ``inplace=True``

        Examples
        --------
        >>> df = cudf.DataFrame({
        ...     "a": [1, 2, 3, 4, 5],
        ...     "b": ["a", "b", "c", "d","e"],
        ...     "c": [1.0, 2.0, 3.0, 4.0, 5.0]
        ... })
        >>> df
           a  b    c
        0  1  a  1.0
        1  2  b  2.0
        2  3  c  3.0
        3  4  d  4.0
        4  5  e  5.0

        Set the index to become the ‘b’ column:

        >>> df.set_index('b')
           a    c
        b
        a  1  1.0
        b  2  2.0
        c  3  3.0
        d  4  4.0
        e  5  5.0

        Create a MultiIndex using columns ‘a’ and ‘b’:

        >>> df.set_index(["a", "b"])
               c
        a b
        1 a  1.0
        2 b  2.0
        3 c  3.0
        4 d  4.0
        5 e  5.0

        Set new Index instance as index:

        >>> df.set_index(cudf.RangeIndex(10, 15))
            a  b    c
        10  1  a  1.0
        11  2  b  2.0
        12  3  c  3.0
        13  4  d  4.0
        14  5  e  5.0

        Setting `append=True` will combine current index with column `a`:

        >>> df.set_index("a", append=True)
             b    c
          a
        0 1  a  1.0
        1 2  b  2.0
        2 3  c  3.0
        3 4  d  4.0
        4 5  e  5.0

        `set_index` supports `inplace` parameter too:

        >>> df.set_index("a", inplace=True)
        >>> df
           b    c
        a
        1  a  1.0
        2  b  2.0
        3  c  3.0
        4  d  4.0
        5  e  5.0
        """

        indexed_df = self._cdf.set_index(keys=keys, drop=drop, append=append, inplace=inplace,
                                         verify_integrity=verify_integrity)
        return DataFrame.from_cudf(indexed_df) if indexed_df else None

    def reset_index(
            self, level=None, drop=False, inplace=False, col_level=0, col_fill=""
    ) -> Union[DataFrame or None]:
        """
        Reset the index.

        Reset the index of the DataFrame, and use the default one instead.

        Parameters
        ----------
        drop : bool, default False
            Do not try to insert index into dataframe columns. This resets
            the index to the default integer index.
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).

        Returns
        -------
        DataFrame or None
            DataFrame with the new index or None if ``inplace=True``.

        Examples
        --------
        >>> df = cudf.DataFrame([('bird', 389.0),
        ...                    ('bird', 24.0),
        ...                    ('mammal', 80.5),
        ...                    ('mammal', np.nan)],
        ...                   index=['falcon', 'parrot', 'lion', 'monkey'],
        ...                   columns=('class', 'max_speed'))
        >>> df
                 class max_speed
        falcon    bird     389.0
        parrot    bird      24.0
        lion    mammal      80.5
        monkey  mammal      <NA>
        >>> df.reset_index()
            index   class max_speed
        0  falcon    bird     389.0
        1  parrot    bird      24.0
        2    lion  mammal      80.5
        3  monkey  mammal      <NA>
        >>> df.reset_index(drop=True)
            class max_speed
        0    bird     389.0
        1    bird      24.0
        2  mammal      80.5
        3  mammal      <NA>
        """
        indexed_df = self._cdf.reset_index(level=level, drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill)
        return DataFrame.from_cudf(indexed_df) if indexed_df else None

    def _convert_subset(self,
                        subset: Union[Hashable, Sequence[Hashable]],
                        ignore_len_check: bool = False) -> Iterable[Hashable]:
        """
        convert the subset to Iterable[Hashable]
        if the any value in subset does not exist in column names, raise an error
        based on: cudf.core.frame.Frame.drop_duplicates

        Returns
        -------
        List/Tuple of column names
        """
        if subset is None:
            subset = self._cdf._column_names
        elif (
            not np.iterable(subset)
            or isinstance(subset, str)
            or isinstance(subset, tuple)
            and subset in self._cdf._data.names
        ):
            subset = (subset,)
        diff = set(subset) - set(self._cdf._data)
        if len(diff) != 0:
            raise ValueError(f"columns {diff} do not exist")

        if ignore_len_check:
            return subset

        if len(subset) == 0:
            raise ValueError("subset size is zero")

        return subset

    def _columns_ok_for_set_ops(self,
                                other: DataFrame,
                                subset: Iterable[Hashable]):
        """
        Check whether:
            other is not None
            other is an instance of DataFrame
            number of columns in both dataframes are the same
            column names in both dataframe are the same (column orders can be different)
            column data types are the same
        """
        if other is None:
            raise ValueError("other can not be null")
        if not isinstance(other, DataFrame):
            raise ValueError("other must be an instance of DataFrame")

        for cname in subset:
            if cname not in other._cdf.columns:
                raise ValueError("other does not have a column named: ", cname)
            # make sure both dataframes have the same data types,
            # columns may be in different order
            if self._cdf.__getattr__(cname).dtype != other._cdf.__getattr__(cname).dtype:
                raise ValueError("column data types are not the same in self and the other dataframe for: ", cname)


    @staticmethod
    def _set_diff(df1: cudf.DataFrame,
                  df2: cudf.DataFrame,
                  subset: Iterable[Hashable]) -> DataFrame:
        """
        Calculate set difference of two local DataFrames
        First calculate a bool mask for the first column. True is both are equal, False otherwise
        Calculate similar bool masks for all columns, and all
        apply the negative of the resulting bool mask to the dataframe
        that gives the difference
        """

        # init bool mask with all True
        bm = [True] * df1._num_rows
        bool_mask = cudf.Series(bm, index=df1.index)

        # determine the rows that are common in all columns
        for cname in subset:
            bool_mask &= df1.__getattr__(cname).isin(df2.__getattr__(cname))

        # get the ones that exist in df1 but not in df2
        diff_df = df1[bool_mask == False]

        return DataFrame.from_cudf(diff_df)

    def set_difference(self,
                       other: DataFrame,
                       subset: Optional[Union[Hashable, Sequence[Hashable]]] = None,
                       env: CylonEnv = None) -> DataFrame:
        """
        set difference operation on two DataFrames

        Parameters
        ----------
        other: second DataFrame to calculate the set difference
        subset: subset of column names to perform the difference,
                if None, use all columns except the index column

        Returns
        -------
        A new DataFrame object constructed by applying set difference operation

        Examples
        --------
        >>> import pygcylon as gc
        >>> df1 = gc.DataFrame({
        ...         'name': ["John", "Smith"],
        ...         'age': [44, 55],
        ... })

        >>> df1
        name  age
        0   John   44
        1  Smith   55
        >>> df2 = gc.DataFrame({
        ...         'age': [44, 66],
        ...         'name': ["John", "Joseph"],
        ... })
        >>> df2
           age    name
        0   44    John
        1   66  Joseph
        >>> df1.set_difference(df2)
            name  age
        1  Smith   55
        >>> df2.set_difference(df1)
           age    name
        1   66  Joseph
        >>> df1.set_difference(df1)
        Empty DataFrame
        Columns: [name, age]
        Index: []

        Works with distributed datafarames similarly
        """

        subset = self._convert_subset(subset=subset)
        self._columns_ok_for_set_ops(other=other, subset=subset)

        df1 = self._cdf
        df2 = other._cdf

        # perform local set difference
        if env is None or env.world_size == 1 or not env.is_distributed:
            return DataFrame._set_diff(df1=df1, df2=df2, subset=subset)

        # column indices for performing distributed shuffle
        df1_hash_columns = [df1._num_indices + df1.columns.to_list().index(cname) for cname in subset]
        df2_hash_columns = [df2._num_indices + df2.columns.to_list().index(cname) for cname in subset]

        shuffled_df1 = _shuffle(df1, hash_columns=df1_hash_columns, env=env)
        shuffled_df2 = _shuffle(df2, hash_columns=df2_hash_columns, env=env)

        return DataFrame._set_diff(df1=shuffled_df1, df2=shuffled_df2, subset=subset)

    def set_union(self,
                  other: DataFrame,
                  subset: Optional[Union[Hashable, Sequence[Hashable]]] = None,
                  keep_duplicates: bool = False,
                  ignore_index: bool = False,
                  env: CylonEnv = None) -> DataFrame:
        """
        set union operation on two DataFrames

        Parameters
        ----------
        other: second DataFrame to calculate the set union
        subset: subset of column names to perform union,
                if None, use all columns except the index column
                used to remove duplicates on the gıven subset of columns
                if keep_duplicates is True, it has no effect
        keep_duplicates: keep the duplicates in the union.
                if True, union is equivalent to concat
        ignore_index: if True, remove old index values and reindex with default index the resulting DataFrame
        env: required for distributed union operation
            has no effect on local operation

        Returns
        -------
        A new DataFrame object constructed by applying set union operation

        Examples
        --------
        >>> import pygcylon as gc
        >>> df1 = gc.DataFrame({
        ...         'name': ["John", "Smith"],
        ...         'age': [44, 55],
        ... })

        >>> df1
        name  age
        0   John   44
        1  Smith   55
        >>> df2 = gc.DataFrame({
        ...         'age': [44, 66],
        ...         'name': ["John", "Joseph"],
        ... })
        >>> df2
           age    name
        0   44    John
        1   66  Joseph
        >>> df1.set_union(df2)
             name  age
        0    John   44
        1  Joseph   66
        1   Smith   55
        >>> df1.set_union(df2, keep_duplicates=True, ignore_index=True)
             name  age
        0    John   44
        1   Smith   55
        2    John   44
        3  Joseph   66

        Works with distributed datafarames similarly
        """

        subset = self._convert_subset(subset=subset)
        self._columns_ok_for_set_ops(other=other, subset=subset)

        concated = concat([self, other], ignore_index=ignore_index)
        return concated if keep_duplicates else concated.drop_duplicates(subset=subset, ignore_index=ignore_index, env=env)

    def set_intersect(self,
                      other: DataFrame,
                      subset: Optional[Union[Hashable, Sequence[Hashable]]] = None,
                      env: CylonEnv = None) -> DataFrame:
        """
        set intersection operation on two DataFrames

        Parameters
        ----------
        other: second DataFrame to calculate the set intersection
        subset: subset of column names to perform intersection,
                if None, use all columns except the index column

        Returns
        -------
        A new DataFrame object constructed by applying set intersection operation

        Examples
        --------
        >>> import pygcylon as gc
        >>> df1 = gc.DataFrame({
        ...         'name': ["John", "Smith"],
        ...         'age': [44, 55],
        ... })

        >>> df1
        name  age
        0   John   44
        1  Smith   55
        >>> df2 = gc.DataFrame({
        ...         'age': [44, 66],
        ...         'name': ["John", "Joseph"],
        ... })
        >>> df2
           age    name
        0   44    John
        1   66  Joseph
        >>> df1.set_intersect(df2)
           name  age
        0  John   44
        >>> df1.set_intersect(df2, subset=["age"])
          name_x  age name_y
        0   John   44   John

        Works with distributed datafarames similarly
        """

        subset = self._convert_subset(subset=subset)
        self._columns_ok_for_set_ops(other=other, subset=subset)

        return self.merge(right=other,
                          how="inner",
                          algorithm="hash",
                          on=subset,
                          left_index=False,
                          right_index=False,
                          sort=False,
                          env=env)

    def shuffle(self, env: CylonEnv, on=None, ignore_index=False, index_shuffle=False) -> DataFrame:
        """
        Shuffle the distributed DataFrame by partitioning 'on' columns.
        If this method is called with a single partition DataFrame, a copy of it is returned.

        Parameters
        ----------
        env: CylonEnv object for this DataFrame
        on: shuffling column name or names as a list
        ignore_index: ignore index when shuffling if True
        index_shuffle: shuffle on index columns if True (on and ingore_index parameters ignored if True)

        Returns
        -------
        A new distributed DataFrame constructed by shuffling the DataFrame
        """
        return comms.shuffle(self, env=env, on=on, ignore_index=ignore_index, index_shuffle=index_shuffle)

    def repartition(self,
                    env: CylonEnv,
                    ignore_index: bool = False,
                    ) -> DataFrame:
        """
        Repartition the dataframe by keeping the global order of rows.
        Repartition the rows evenly among the workers
        For example:
            if there are 4 partitions currently with row counts: [10, 20 ,30 ,40]
            After repartitioning evenly, row counts become: [25, 25 ,25 ,25]

            if the number of partitions is not a multiple of sum of row counts,
              first k partitions get one more row:
              if there are 4 partitions currently with row counts: [10, 20 ,30 ,42]
              After repartitioning evenly, row counts become: [26, 26 ,25 ,25]

        Parameters
        ----------
        ignore_index: ignore index when repartitioning if True
        env: CylonEnv object for this DataFrame

        Returns
        -------
        A new distributed DataFrame constructed by repartitioning the DataFrame
        """
        return comms.repartition(self, env=env, ignore_index=ignore_index)

    def allgather(self,
                  env: CylonEnv,
                  ignore_index: bool = False,
                  ) -> DataFrame:
        """
        AllGather all dataframe partitions to all workers by keeping the global order of rows.
        For example:
            if there are 4 partitions currently with row counts: [10, 20 ,30 ,40]
            After allgathering all partitions to all workers,
            a new distributed dataframe is constructed with row counts: [100, 100 ,100 ,100]

        Parameters
        ----------
        env: CylonEnv object for this DataFrame
        ignore_index: ignore index when allgathering if True

        Returns
        -------
        A new distributed DataFrame constructed by allgathering all distributed dataframes to all workers
        """
        return comms.allgather(self, env=env, ignore_index=ignore_index)

    def equals(self, other, **kwargs):
        """
        Test whether two objects contain the same elements.
        This function allows two Series or DataFrames to be compared against
        each other to see if they have the same shape and elements. NaNs in
        the same location are considered equal. The column headers do not
        need to have the same type.

        This method performs local equality only.
        Todo: We need to add distributed equality comparison.

        Parameters
        ----------
        other : Series or DataFrame
            The other Series or DataFrame to be compared with the first.

        Returns
        -------
        bool
            True if all elements are the same in both objects, False
            otherwise.

        Examples
        --------
        >>> import pygcylon as gcy

        Comparing DataFrames with `equals`:

        >>> df = gcy.DataFrame({1: [10], 2: [20]})
        >>> df
            1   2
        0  10  20
        >>> exactly_equal = gcy.DataFrame({1: [10], 2: [20]})
        >>> exactly_equal
            1   2
        0  10  20
        >>> df.equals(exactly_equal)
        True

        For two DataFrames to compare equal, the types of column
        values must be equal, but the types of column labels
        need not:

        >>> different_column_type = gcy.DataFrame({1.0: [10], 2.0: [20]})
        >>> different_column_type
           1.0  2.0
        0   10   20
        >>> df.equals(different_column_type)
        True
        """
        return self._cdf.equals(other=other.to_cudf(), **kwargs)

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=False,
        group_keys=True,
        squeeze=False,
        observed=False,
        dropna=True,
        env: CylonEnv = None
    ) -> GroupByDataFrame:
        """
        Group DataFrame using a mapper or by a Series of columns.
        Works with both single DataFrame or distributed DataFrame

        A groupby operation involves some combination of splitting the object,
        applying a function, and combining the results. This can be used to
        group large amounts of data and compute operations on these groups.

        When calculating groupby in distributed DataFrames,
        an all-to-all shuffle communication operation is performed.
        It is very important to avoid unnecessary shuffle operations,
        since the shuffling of the dataframe among distributed workers are constly.

        When a GroupByDataFrame object is created, and the first groupby operation is performed,
        this shuffle operation is performed by partitioning the tables on the groupby columns and
        all dataframe columns are shuffled.

        So, to get the best performance in a distributed dataframe,
        one should first create a GroupByDataFrame object and perform many aggregations on it.
        Because, creating and performing a groupby object requires a distributed shuffle.
        When we reuse the same GroupByDataFrame object, we avoid re-shuffling the dataframe.
        For example following code performs a single shuffle only:
            gby = df.groupby(["column1", "column2"], ..., env=env)
            gby.sum()
            gby["columnx"].mean()
            gby[["columnx", "columny"]].min()

        One must avoid running the groupby operation on the dataframe object.
        For example, all three of the following operations perform the a separate distributed shuffle:
            df.groupby("columnq", env=env)["columnb"].sum()
            df.groupby("columnq", env=env)["columnb"].mean()
            df.groupby("columnq", env=env)["columnc"].max()
        One can easily perform a single shuffle for these three lines by first creating a GroupByDataFrame object
        and performing the aggragations using it.

        A second important point is to create a new dataframe from a subset of columns
        and performing the groupby on it when working with dataframes with many columns.
        Suppose, you are working with a dataframe with hundreds of columns
        but you would like to perform the groupby and aggregations on a small number of columns.
        First, you need to create a new dataframe with those groupby and aggregations columns.
        Then, perform the groupby on this new dataframe.
        This will avoid shufling the whole dataframe. Only the columns on the new dataframe will be shuffled.
            df2 = df[["columnx", "columny", "columnz", ...]]
            gby = df2.groupby("columnx", env=env)
            gby["columny"].sum()
            gby["columnz"].mean()
        In this case, the shuffling is performed only on the columns of df2.


        Parameters
        ----------
        by : mapping, function, label, or list of labels
            Used to determine the groups for the groupby. If by is a
            function, it’s called on each value of the object’s index.
            If a dict or Series is passed, the Series or dict VALUES will
            be used to determine the groups (the Series’ values are first
            aligned; see .align() method). If a cupy array is passed, the
            values are used as-is determine the groups. A label or list
            of labels may be passed to group by the columns in self.
            Notice that a tuple is interpreted as a (single) key.
        level : int, level name, or sequence of such, default None
            If the axis is a MultiIndex (hierarchical), group by a particular
            level or levels.
        as_index : bool, default True
            For aggregated output, return object with group labels as
            the index. Only relevant for DataFrame input.
            as_index=False is effectively “SQL-style” grouped output.
        sort : bool, default False
            Sort result by group key. Differ from Pandas, cudf defaults to
            ``False`` for better performance. Note this does not influence
            the order of observations within each group. Groupby preserves
            the order of rows within each group.
        dropna : bool, optional
            If True (default), do not include the "null" group.
        env: CylonEnv needs to be provided for distributed groupby operation.

        Returns
        -------
            DataFrameGroupBy
                Returns a groupby object that contains information
                about the groups.

        Examples
        --------
        >>> import pygcylon as gc
        >>> # first try local groupby on a single DataFrame
        >>> df = gc.DataFrame({'a': [1, 1, 1, 2, 2], 'b': [1, 1, 2, 2, 3], 'c': [1, 2, 3, 4, 5]})
        >>> df
           a  b  c
        0  1  1  1
        1  1  1  2
        2  1  2  3
        3  2  2  4
        4  2  3  5
        >>> # create a groupby object and perform multiple operations
        >>> gby = df.groupby("a")
        >>> gby.sum()
           b  c
        a
        2  5  9
        1  4  6
        >>> gby.max()
           b  c
        a
        2  3  5
        1  2  3
        >>> gby["b"].sum()
           b
        a
        2  5
        1  4
        >>> # to perform groupby on a different set of columns, we need to create a new GroupByDataFrame object
        >>> gby = df.groupby(["a", "b"])
        >>> gby.sum()
             c
        a b
        1 2  3
        2 2  4
          3  5
        1 1  3

        >>> # todo: add distributed DataFrame examples
        >>> env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)

        """
        if axis not in (0, "index"):
            raise NotImplementedError("axis parameter is not yet implemented")

        if group_keys is not True:
            raise NotImplementedError(
                "The group_keys keyword is not yet implemented"
            )

        if squeeze is not False:
            raise NotImplementedError(
                "squeeze parameter is not yet implemented"
            )

        if observed is not False:
            raise NotImplementedError(
                "observed parameter is not yet implemented"
            )

        if by is None and level is None:
            raise TypeError(
                "groupby() requires either by or level to be specified."
            )

        return GroupByDataFrame(
            self,
            by=by,
            level=level,
            as_index=as_index,
            dropna=dropna,
            sort=sort,
            env=env,
        )

    @staticmethod
    def _get_column_indices(df: cudf.DataFrame, column_names: List[str], ignore_index=True) -> List[int]:
        """
        get column indices from column names
        """
        if not isinstance(column_names, list) and not isinstance(column_names[0], str):
            raise ValueError

        number_of_index_columns = 0 if ignore_index else df._num_indices
        return [number_of_index_columns + df.columns.to_list().index(cname) for cname in column_names]

    def sort_index(
            self,
            axis=0,
            level=None,
            ascending=True,
            inplace=False,
            kind=None,
            na_position="last",
            sort_remaining=True,
            ignore_index=False,
            env: CylonEnv = None,
    ):
        """Sort distributed DataFrame by labels (along an axis).

        Parameters
        ----------
        axis : {0 or ‘index’, 1 or ‘columns’}, default 0
            The axis along which to sort. The value 0 identifies the rows,
            and 1 identifies the columns.
            only 0 is supported.
        level : int or level name or list of ints or list of level names
            If not None, sort on values in specified index level(s).
            This is only useful in the case of MultiIndex.
        ascending : bool, default True
            Sort ascending vs. descending.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : sorting method such as `quick sort` and others.
            Not yet supported.
        na_position : {‘first’, ‘last’}, default ‘last’
            Puts NaNs at the beginning if first; last puts NaNs at the end.
        sort_remaining : bool, default True
            Not yet supported
        ignore_index : bool, default False
            if True, index will be replaced with RangeIndex.

        Returns
        -------
        DataFrame or None

        """
        if env is None or env.world_size == 1:
            sorted_cdf = self._cdf.sort_index(axis=axis,
                                              level=level,
                                              ascending=ascending,
                                              inplace=inplace,
                                              kind=kind,
                                              na_position=na_position,
                                              sort_remaining=sort_remaining,
                                              ignore_index=ignore_index)
            return DataFrame.from_cudf(sorted_cdf)

        if kind is not None:
            raise NotImplementedError("kind is not yet supported")

        if level is not None:
            raise NotImplementedError("level is not yet supported")

        if not sort_remaining:
            raise NotImplementedError(
                "sort_remaining == False is not yet supported"
            )
        if axis not in (0, "index"):
            raise NotImplementedError("can sort only row wise")

        nulls_after = True if na_position == "last" else False
        sorted_tbl = distributed_sort(self._cdf,
                                      context=env.context,
                                      ascending=ascending,
                                      nulls_after=nulls_after,
                                      ignore_index=False,
                                      by_index=True)
        sorted_cdf = cudf.DataFrame._from_data(sorted_tbl._data, sorted_tbl._index)

        if ignore_index is True:
            sorted_cdf = sorted_cdf.reset_index(drop=True)

        return DataFrame.from_cudf(sorted_cdf)

    def sort_values(
            self,
            by,
            axis=0,
            ascending=True,
            inplace=False,
            na_position="last",
            ignore_index=False,
            env: CylonEnv = None
    ) -> DataFrame:
        """
        sort dataframe based on given columns
        Perform distributed sorting on the dataframe.
        When DataFrame is sorted in ascending order, After the sorting:
            First worker will have the values: [min to smallest(k)]
            Second worker will have the values: [smallest(k+1) to smallest(l)]
            Third worker will have the values: [smallest(l+1) to smallest(m)]
            ....
            Last worker will have the values: [largest(p) to largest(n)]
        where 0 < k < l < m < p < n

        When the DataFrame is sorted in descending order,
        first worker will have the largest values and the last worker will have the smallest values

        Distributed Sorting Algorithm:
        1. Each worker sorts its local DataFrame
        2. Each worker samples the sorted dataframe uniformly and sends the samples to the first worker
        3. First workers receives all samples, merges sub sorted samples and gets all samples sorted
        4. First worker determines global split points for the ultimate sorted array based on samples.
        5. First worker broadcasts the split points to all workers
        6. All workers sends the corresponding sorted sub arrays to other workers
        7. Each worker merges sorted sub-ranges from all other workers

        Parameters
        ----------
        by : str or list of str
            Name or list of names to sort by.
        axis : only column based sorting supported with axis=0
        ascending : bool or list of bool, default True
            Sort ascending vs. descending. Specify list for multiple sort
            orders. If this is a list of bools, must match the length of the
            by.
        inplace: no inplace sorting
        na_position : {‘first’, ‘last’}, default ‘last’
            'first' puts nulls at the beginning, 'last' puts nulls at the end
        ignore_index : bool, default False
            If True, index will not be sorted.
        env : CylonEnv object for distributed sorting

        """
        if inplace:
            raise NotImplementedError("`inplace` not currently implemented.")
        if axis != 0:
            raise NotImplementedError("`axis` not currently implemented.")
        if isinstance(by, list) and isinstance(ascending, list) and len(by) != len(ascending):
            raise ValueError("size of 'by' and 'ascending' lists must match.")
        if not (isinstance(ascending, bool) or isinstance(ascending, list)):
            raise ValueError("ascending must be either bool or list of bool.")

        if env is None or env.world_size == 1:
            sorted_cdf = self._cdf.sort_values(by=by,
                                               ascending=ascending,
                                               inplace=inplace,
                                               na_position=na_position,
                                               ignore_index=ignore_index)
            return DataFrame.from_cudf(sorted_cdf)

        # make sure by is a list of string
        if isinstance(by, str):
            by = [by]
        if not isinstance(by, list):
            raise ValueError("by has to be either str or list[str]")

        nulls_after = True if na_position == "last" else False
        sorted_tbl = distributed_sort(self._cdf,
                                      context=env.context,
                                      sort_columns=by,
                                      ascending=ascending,
                                      nulls_after=nulls_after,
                                      ignore_index=ignore_index,
                                      by_index=False)
        sorted_cdf = cudf.DataFrame._from_data(sorted_tbl._data, sorted_tbl._index)
        return DataFrame.from_cudf(sorted_cdf)

    def row_counts_for_all(self, env: CylonEnv = None) -> List[int]:
        """
        Get the number of rows in all DataFrame partitions
        In the returned list,
            first element shows the number_of_rows in the first worker partition
            second element shows the number_of_rows in the second worker partition
            third element shows the number_of_rows in the third worker partition
            ....
            last element shows the number_of_rows in the last worker partition
        """

        if env is None or env.world_size == 1:
            return [len(self._cdf._index)]

        num_rows = len(self._cdf)
        return row_counts_all_tables(num_rows=num_rows, context=env.context)

    @staticmethod
    def _calculate_row_counts_head_tail(n: int, head: bool, all_row_counts: List[int]) -> List[int]:
        """
        calculate the distribution of rows to workers in head/tail DataFrame
        """

        nworkers = len(all_row_counts)
        indices = [i for i in range(nworkers)] if head else [i for i in reversed(range(nworkers))]

        row_counts = [0] * nworkers
        total = 0
        for i in indices:
            if total + all_row_counts[i] < n:
                row_counts[i] = all_row_counts[i]
                total += all_row_counts[i]
            elif total + all_row_counts[i] >= n:
                row_counts[i] = n - total
                break

        return row_counts

    def head(self, n, env: CylonEnv = None) -> DataFrame:
        """
        Return a new DataFrame with top n rows in it.
        If the first worker has more than n rows,
            all rows will be returned from the first worker.
            Top n rows of the first worker will be returned.
        If the first worker has less than n rows, then:
            rows from the second worker will be returned.
        If the second worker also does not have enough rows,
            rows from the third worker will be returned.
        etc.

        The resulting DataFrame object may have rows in the first worker only,
        or it may have rows in other workers as well.
        This function does not bring all rows back to the first worker.
        Rows will be residing in the worker where they are copied from.
        For example:
            If a DataFrame has 4 partitions with following row counts: [50, 60, 20, 30]
            DataFrame returned by head(20) will have rows: [20, 0, 0, 00]
            DataFrame returned by head(60) will have rows: [50, 10, 0, 0]
            DataFrame returned by head(120) will have rows: [50, 60, 10, 0]
            DataFrame returned by head(1000) will have rows: [50, 60, 20, 30]

        If the user wants to collect all rows to a worker,
        a separate function should be executed.

        Parameters
        ----------
        n: the number of rows to return
        env: Cylon environment object. Required for distributed head operation.

        Returns
        -------
        A new DataFrame object constructed by head N rows.
        """

        if env is None or env.world_size == 1:
            head_n_cdf = self._cdf.head(n)
            return DataFrame.from_cudf(head_n_cdf)

        row_counts = self.row_counts_for_all(env=env)
        head_row_counts = DataFrame._calculate_row_counts_head_tail(n=n, head=True, all_row_counts=row_counts)

        head_n_cdf = self._cdf.head(head_row_counts[env.rank])
        return DataFrame.from_cudf(head_n_cdf)

    def tail(self, n, env: CylonEnv = None) -> DataFrame:
        """
        Return a new DataFrame with tail n rows in it.
        If the last worker has more than n rows,
            all rows will be returned from the last worker.
            Tail n rows of the last worker will be returned.
        If the last worker has less than n rows, then:
            rows from the (last-1) worker will be returned.
        If the (last-1) worker also does not have enough rows,
            rows from the (last-2) worker will be returned.
        etc.

        The resulting DataFrame object may have rows in the last worker only,
        or it may have rows in other workers as well.
        This function leaves the rows where they have been residing.
        For example:
            If a DataFrame has 4 partitions with following row counts: [50, 60, 20, 30]
            DataFrame returned by tail(20) will have rows: [0, 0, 0, 20]
            DataFrame returned by tail(40) will have rows: [0, 0, 10, 30]
            DataFrame returned by tail(100) will have rows: [0, 50, 20, 30]
            DataFrame returned by tail(1000) will have rows: [50, 60, 20, 30]

        If the user wants to collect all rows to a worker,
        a separate function should be executed.

        Parameters
        ----------
        n: the number of rows to return
        env: Cylon environment object. Required for distributed head operation.

        Returns
        -------
        A new DataFrame object constructed by head N rows.
        """

        if env is None or env.world_size == 1:
            tail_n_cdf = self._cdf.tail(n)
            return DataFrame.from_cudf(tail_n_cdf)

        row_counts = self.row_counts_for_all(env=env)
        tail_row_counts = DataFrame._calculate_row_counts_head_tail(n=n, head=False, all_row_counts=row_counts)

        tail_n_cdf = self._cdf.head(tail_row_counts[env.rank])
        return DataFrame.from_cudf(tail_n_cdf)

    def to_csv(self, file_names, env: CylonEnv = None, **kwargs) -> str:
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
        return pygcylon.io.write_csv(self, file_names=file_names, env=env, **kwargs)

    def to_json(self, file_names, env: CylonEnv = None, **kwargs) -> str:
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
        return pygcylon.io.write_json(self, file_names=file_names, env=env, **kwargs)

    def to_parquet(self, file_names,
                   env: CylonEnv = None,
                   write_metadata_file: bool = True,
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
        write_metadata_file: write metadata to _metadata file
        kwargs: the parameters that will be passed on to cudf.DataFrame.to_json function

        Returns
        -------
        Filename written
        """
        return pygcylon.io.write_parquet(self, file_names=file_names,
                                         env=env,
                                         write_metadata_file=write_metadata_file,
                                         **kwargs)


def concat(
        dfs,
        axis=0,
        join="outer",
        ignore_index: bool = False,
        keys=None,
        levels=None,
        names=None,
        verify_integrity: bool = False,
        sort: bool = False,
        copy: bool = True,
        env: CylonEnv = None
) -> DataFrame:
    """Concatenate DataFrames row-wise.

    Parameters
    ----------
    dfs: list of DataFrames to concatenate
    axis: {0/'index', 1/'columns'}, default 0
        The axis to concatenate along. (Currently only 0 supported)
    join: {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis (or axes).
    ignore_index: bool, default False
        Set True to ignore the index of the *dfs* and provide a
        default range index instead.
    keys (Unsupported) : sequence, default None
        If multiple levels passed, should contain tuples. Construct
        hierarchical index using the passed keys as the outermost level.
    levels (Unsupported) : list of sequences, default None
        Specific levels (unique values) to use for constructing a
        MultiIndex. Otherwise they will be inferred from the keys.
    names (Unsupported) : list, default None
        Names for the levels in the resulting hierarchical index.
    verify_integrity (Unsupported) : bool, default False
        Check whether the new concatenated axis contains duplicates. This can
        be very expensive relative to the actual data concatenation.
    sort : bool, default False
        Sort non-concatenation axis if it is not already aligned when `join`
        is 'outer'.
        This has no effect when ``join='inner'``, which already preserves
        the order of the non-concatenation axis.
    copy (Unsupported) : bool, default True
        If False, do not copy data unnecessarily.
    env: Cylon environment object

    Returns
    -------
    A new DataFrame object constructed by concatenating all input DataFrame objects.

    Examples
    --------

    Combine two ``DataFrame`` objects with identical columns.

    >>> df1 = DataFrame([['a', 1], ['b', 2]],
    ...                    columns=['letter', 'number'])
    >>> df1
    letter  number
    0      a       1
    1      b       2
    >>> df2 = DataFrame([['c', 3], ['d', 4]],
    ...                    columns=['letter', 'number'])
    >>> df2
    letter  number
    0      c       3
    1      d       4
    >>> DataFrame.concat([df1, df2])
    letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` objects with overlapping columns
    and return everything. Columns outside the intersection will
    be filled with ``NaN`` values.

    >>> df3 = DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']],
    ...                    columns=['letter', 'number', 'animal'])
    >>> df3
    letter  number animal
    0      c       3    cat
    1      d       4    dog
    >>> DataFrame.concat([df1, df3], sort=False)
    letter  number animal
    0      a       1    NaN
    1      b       2    NaN
    0      c       3    cat
    1      d       4    dog

    Combine ``DataFrame`` objects with overlapping columns
    and return only those that are shared by passing ``inner`` to
    the ``join`` keyword argument.

    >>> DataFrame.concat([df1, df3], join="inner")
    letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    (Unsupported) Combine ``DataFrame`` objects horizontally along the x axis by
    passing in ``axis=1``.

    >>> df4 = DataFrame([['bird', 'polly'], ['monkey', 'george']],
    ...                    columns=['animal', 'name'])
    >>> DataFrame.concat([df1, df4], axis=1)

    letter  number  animal    name
    0      a       1    bird   polly
    1      b       2  monkey  george

    (Unsupported) Prevent the result from including duplicate index values with the
    ``verify_integrity`` option.

    >>> df5 = DataFrame([1], index=['a'])
    >>> df5
    0
    a  1
    >>> df6 = DataFrame([2], index=['a'])
    >>> df6
    0
    a  2
    >>> DataFrame.concat([df5, df6], verify_integrity=True)
    Traceback (most recent call last):
        ...
    ValueError: Indexes have overlapping values: ['a']
    """

    if not dfs:
        raise ValueError("No DataFrames to concatenate")

    # remove None objects if any
    dfs = [obj for obj in dfs if obj is not None]
    if len(dfs) == 0:
        raise ValueError("No DataFrames to concatenate after None removal")

    if axis != 0:
        raise ValueError("Only concatenation on axis 0 is currently supported")

    if verify_integrity not in (None, False):
        raise NotImplementedError("verify_integrity parameter is not supported yet.")

    if keys is not None:
        raise NotImplementedError("keys parameter is not supported yet.")

    if levels is not None:
        raise NotImplementedError("levels parameter is not supported yet.")

    if names is not None:
        raise NotImplementedError("names parameter is not supported yet.")

    if not copy:
        raise NotImplementedError("copy can be only True.")

    # make sure all dfs DataFrame objects
    for obj in dfs:
        if not isinstance(obj, DataFrame):
            raise ValueError("Only DataFrame objects can be concatenated")

    # perform local concatenation, no need to distributed concat
    dfs = [obj._cdf for obj in dfs]
    concated_df = cudf.concat(dfs, axis=axis, join=join, ignore_index=ignore_index, sort=sort)
    return DataFrame.from_cudf(concated_df)


def _shuffle(df: cudf.DataFrame, hash_columns, env: CylonEnv = None, ignore_index=False) -> cudf.DataFrame:
    """
    Perform shuffle on a distributed dataframe
    :param df: local cudf DataFrame object
    :param hash_columns: column indices to partition the table
    :param env: CylonEnv
    :ignore_index: ignore index when shuffling
    :return: shuffled dataframe as a new object
    """
    if env is None:
        raise ValueError("No CylonEnv is given.")
    if env.world_size == 1:
        raise ValueError(f"Not a distributed DataFrame. No shuffling for local DataFrames.")

    tbl = cshuffle(df, hash_columns=hash_columns, ignore_index=ignore_index, context=env.context)
    return cudf.DataFrame._from_data(tbl._data, tbl._index)
