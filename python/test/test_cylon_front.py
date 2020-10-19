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
Running the test:
>>> python python/test/test_cylon_front.py --table1_path /tmp/user_usage_tm_1.csv --table2_path /tmp/user_device_tm_1.csv
'''

from pycylon import Table
from pycylon import CylonContext
from pycylon.net import MPIConfig
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
import argparse

mpi_config = MPIConfig()
ctx: CylonContext = CylonContext(config=mpi_config, distributed=True)

parser = argparse.ArgumentParser(description='PyCylon Table Conversion')
parser.add_argument('--table1_path', type=str, help='Path to table 1 csv')
parser.add_argument('--table2_path', type=str, help='Path to table 2 csv')

args = parser.parse_args()

csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

tb1: Table = read_csv(ctx, args.table1_path, csv_read_options)

tb2: Table = read_csv(ctx, args.table2_path, csv_read_options)

configs = {'join_type': 'inner', 'algorithm': 'sort'}

tb3: Table = tb1.distributed_join(table=tb2,
                      join_type=configs['join_type'],
                      algorithm=configs['algorithm'],
                      left_on=[3],
                      right_on=[0]
                      )

tb3.show()

ctx.finalize()
