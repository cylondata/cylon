#print("From Imports : 1")
import ctypes
import os
#ctypes.cdll.LoadLibrary('/home/vibhatha/github/forks/twisterx/python/twisterx/lib/libarrow.so.16')
## https://stackoverflow.com/questions/2329722/nm-u-the-symbol-is-undefined
#ctypes.cdll.LoadLibrary('/home/vibhatha/github/forks/twisterx/python/twisterx/lib/libtwisterx.so')
from pytwisterx import tablebuilder
from pytwisterx.common.status import Status
# print("Import Completed")
#
print(tablebuilder.id(), tablebuilder.rows(), tablebuilder.columns())
print("Print Tables")

path =b'/tmp/csv.csv'
id = b'id'

s: Status = tablebuilder.csv(b"s", b"s")
print("Status got from reading")
print(s.get_code())

# _ZN8twisterx2io8read_csvERKSs


