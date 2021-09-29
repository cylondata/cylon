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
>>  mpirun --mca opal_cuda_support 1 -n 4 -quiet python -m pytest --with-mpi -q python/pygcylon/test/test_sort.py
'''

import pytest
import cudf
import pycylon as cy
import pygcylon as gcy


@pytest.mark.mpi
def test_sort_by_index():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    input_file = "data/gather/sales_records_nulls_nunascii_" + str(env.rank) + ".csv"
    sorted_file = "data/gather/sales_index_sorted_" + str(env.rank) + ".csv"

    df = gcy.DataFrame.from_cudf(cudf.read_csv(input_file,
                                               parse_dates=["Order Date"],
                                               infer_datetime_format=True))
    df = df.set_index(keys="Order Date")
    index_sorted = df.sort_index(env=env)

    saved_sorted_df = gcy.DataFrame.from_cudf(cudf.read_csv(sorted_file,
                                                            index_col="Order Date",
                                                            parse_dates=["Order Date"],
                                                            infer_datetime_format=True))
    assert index_sorted.equals(saved_sorted_df), \
        "Index sorted DataFrame and DataFrame from file are not equal"

#    env.finalize()


@pytest.mark.mpi
def test_sort_by_value_numeric():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    input_file = "data/gather/numeric_" + str(env.rank) + ".csv"
    sorted_file = "data/gather/numeric_sorted_" + str(env.rank) + ".csv"

    df = gcy.DataFrame.from_cudf(cudf.read_csv(input_file))
    sorted_df = df.sort_values(by=["0", "1"], ignore_index=True, env=env)

    saved_sorted_df = gcy.DataFrame.from_cudf(cudf.read_csv(sorted_file))
    assert sorted_df.equals(saved_sorted_df), \
        "Numeric value sorted DataFrame and DataFrame from file are not equal"

#    env.finalize()

@pytest.mark.mpi
def test_sort_by_value_all():
    env: cy.CylonEnv = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
    print("CylonContext Initialized: My rank: ", env.rank)

    input_file = "data/gather/sales_records_nulls_nunascii_" + str(env.rank) + ".csv"
    sorted_file = "data/gather/sales_sorted_" + str(env.rank) + ".csv"

    df = gcy.DataFrame.from_cudf(cudf.read_csv(input_file,
                                               parse_dates=["Order Date"],
                                               infer_datetime_format=True))

    sorted_df = df.sort_values(by=["Country", "Item Type"], ignore_index=True, env=env)

    saved_sorted_df = gcy.DataFrame.from_cudf(cudf.read_csv(sorted_file,
                                                            parse_dates=["Order Date"],
                                                            infer_datetime_format=True))

    assert sorted_df[["Country", "Item Type"]].equals(saved_sorted_df[["Country", "Item Type"]]), \
        "Numeric value sorted DataFrame and DataFrame from file are not equal"

#    env.finalize()
