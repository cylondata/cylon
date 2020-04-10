import ctypes

#ctypes.cdll.LoadLibrary('/home/vibhatha/github/forks/twisterx/cpp/build/arrow/install/lib/libarrow.so')
ctypes.cdll.LoadLibrary('/home/vibhatha/github/forks/twisterx/cpp/build/lib/libtwisterx.so')

from pytwisterx import tablebuilder
from pytwisterx.common.status import Status


print(tablebuilder.id(), tablebuilder.rows(), tablebuilder.columns())

path =b'path'
id = b'id'

s: Status = tablebuilder.csv(path, id)

print(s.get_code())

# _ZN8twisterx2io8read_csvERKSs


