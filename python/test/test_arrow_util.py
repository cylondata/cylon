from pytwisterx.arrow.util import ArrowUtil
import pyarrow as pa
import numpy as np
arr = pa.array([1,2,3,4,5])
len = ArrowUtil.get_array_length(arr)
print("Array length {} ".format(len))

from pyarrow import csv

fn = '/tmp/csv.csv'

table = csv.read_csv(fn)

print(table)

ArrowUtil.get_table_info(table)


