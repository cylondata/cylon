from typing import List

# class Table(object):
#
#     def __init__(self, data: List = None):
#         self._tb_data = data
#
#     def execute(self, func):
#         for datum in self._tb_data:
#             func(datum)
#
#
# def udf1(datum=None):
#     print(datum)
#
#
# tb = Table([1, 2, 3, 4, 5])
#
# tb.execute(udf1)

from pyarrow.compute import add


def addition(n):
    return add(n, n)


# We double all numbers using map()
import pyarrow as pa

a = pa.array([1, 2, 3, 4])
numbers = (1, 2, 3, 4)

result = map(addition, a)
rl = list(result)
print(rl[0], type(rl[0]))
