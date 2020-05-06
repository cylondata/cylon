import time
from pytwisterx.arrow.util import ArrowUtil
import pyarrow as pa
import numpy as np
arr = pa.array([1,2,3,4,5])
len = ArrowUtil.get_array_length(arr)
print("Array length {} ".format(len))

ArrowUtil.get_array_info(arr)

from pyarrow import csv

fn = '/tmp/csv.csv'

table = csv.read_csv(fn)

print(table)

ArrowUtil.get_table_info(table)

t1 = time.time_ns()
table.to_pandas()
t2 = time.time_ns()



print("Time Taken For Pandas Conversion : {}".format(t2-t1))






