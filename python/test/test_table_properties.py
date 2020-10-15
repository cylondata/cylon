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

# from pycylon import Table
# from pycylon.csv import csv_reader
# from pycylon import CylonContext
#
# ctx: CylonContext = CylonContext(config=None)
#
# tb: Table = csv_reader.read(ctx, '/tmp/user_usage_tm_1.csv', ',')
#
# print("Table Column Names")
# print(tb.column_names)
#
# print("Table Schema")
# print(tb.schema)
#
# print(tb[0].to_pandas())
#
# print(tb[0:5].to_pandas())
#
# print(tb[2:5].to_pandas())
#
# print(tb[5].to_pandas())
#
# print(tb[7].to_pandas())
#
# tb.show_by_range(0, 4, 0, 4)
#
# print(tb[0:5].to_pandas())

ctx.finalize()

import pyarrow as pa

arw_table: pa.Table = tb.to_arrow()

