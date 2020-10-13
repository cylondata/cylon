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

import argparse
import pyarrow as pa
from pyarrow import csv
from pycylon import Table
from pycylon import CylonContext

'''
running test case 
>>> python test/test_pyarrow.py --table_path /tmp/csv.csv 
'''


parser = argparse.ArgumentParser(description='PyCylon Table Conversion')
parser.add_argument('--table1_path', type=str, help='Path to table 1 csv')

args = parser.parse_args()

ctx: CylonContext = CylonContext(config=None)

tb = csv.read_csv(args.table1_path)

print(tb)

tb = Table.from_arrow(tb, ctx)

print(ctx.get_world_size())

tb.show()

ctx.finalize()