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
from pycylon import Table
from pycylon.csv import csv_reader

if __name__ == "__main__":
    ctx: CylonContext = CylonContext("mpi")

    csv1 = "/tmp/csv1.csv"
    csv2 = "/tmp/csv2.csv"

    first_table: Table = csv_reader.read(ctx, csv1, ',')
    second_table: Table = csv_reader.read(ctx, csv2, ',')

    try:
        print(f"Table 1 & 2 Rows [{first_table.rows},{second_table.rows}], "
              f"Columns [{first_table.columns},{first_table.columns}]")
    except Exception:
        raise Exception("Something went wrong in loading tables from disk")

    joined_table: Table = first_table.distributed_join(ctx, table=second_table, join_type="inner",
                                                       algorithm="sort",
                                                       left_col=0, right_col=0)
    print(f"First table had : {first_table.rows} and Second table had : {second_table.rows}, "
          f"Joined has : {joined_table.rows}")

    ctx.finalize()
