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

from pycylon import CylonContext
from pycylon import Table, CylonEnv
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
from pycylon.net import MPIConfig

if __name__ == "__main__":


    env = CylonEnv(config=MPIConfig(), distributed=True)
    csv_read_options = CSVReadOptions() \
        .use_threads(False) \
        .block_size(1 << 30) \
        .na_values(['na', 'none'])

    rank = env.rank + 1

    print(f"rank: {rank}")
    csv1 = f"/home/qad5gv/cylon/data/input/user_usage_tm_{rank}.csv"
    csv2 = f"/home/qad5gv/cylon/data/input/user_device_tm_{rank}.csv"

    first_table: Table = read_csv(env, csv1, csv_read_options)
    second_table: Table = read_csv(env, csv2, csv_read_options)

    print(first_table)

    first_row_count = first_table.row_count

    second_row_count = second_table.row_count


    print(f"Table 1 & 2 Rows [{first_row_count},{second_row_count}], "
         f"Columns [{first_table.column_count},{second_table.column_count}]")

    configs = {'join_type': 'inner', 'algorithm': 'sort'}
    joined_table: Table = first_table.distributed_join(table=second_table,
                                      join_type=configs['join_type'],
                                      algorithm=configs['algorithm'],
                                      left_on=[3],
                                      right_on=[0]
                                      )



    join_row_count = joined_table.row_count

    print(f"First table had : {first_row_count} and Second table had : {second_row_count}, "
          f"Joined has : {join_row_count}")


    env.finalize()
