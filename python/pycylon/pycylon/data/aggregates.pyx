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

from pycylon.data.aggregates cimport *
from pycylon.common.status cimport CStatus
from pycylon.api.lib cimport pycylon_unwrap_context, pycylon_unwrap_table, pycylon_wrap_column

from pycylon.data.table import Table
from pycylon.ctx.context import CylonContext
from pycylon.data.column import Column

cpdef enum AggregationOp:
    SUM = CGroupByAggregationOp.CSUM
    COUNT = CGroupByAggregationOp.CCOUNT
    MIN = CGroupByAggregationOp.CMIN
    MAX = CGroupByAggregationOp.CMAX
    VAR = CGroupByAggregationOp.CVAR
    NUNIQUE = CGroupByAggregationOp.CNUNIQUE
    MEAN = CGroupByAggregationOp.CMEAN
    QUANTILE = CGroupByAggregationOp.CQUANTILE
    STDDEV = CGroupByAggregationOp.CSTDDEV

AggregationOpString = {
    'sum': CGroupByAggregationOp.CSUM,
    'cnt': CGroupByAggregationOp.CCOUNT,
    'min': CGroupByAggregationOp.CMIN,
    'max': CGroupByAggregationOp.CMAX,
    'var': CGroupByAggregationOp.CVAR,
    'nunique': CGroupByAggregationOp.CNUNIQUE,
    'mean': CGroupByAggregationOp.CMEAN,
    'quantile': CGroupByAggregationOp.CQUANTILE,
    'std': CGroupByAggregationOp.CSTDDEV,
}

def sum_table(ctx: CylonContext, table: Table, skipna=True) -> Column:
    cdef shared_ptr[CCylonContext] cctx = pycylon_unwrap_context(ctx)
    cdef shared_ptr[CTable] ctable = pycylon_unwrap_table(table)
    cdef CBasicOptions options = CBasicOptions(skipna)
    cdef shared_ptr[CColumn] cresult
    cdef CStatus status = SumTable(cctx, ctable, &cresult, options)

    if status.is_ok():
        return pycylon_wrap_column(cresult)
    else:
        raise Exception(f"aggregation error: {status.get_msg().decode()}")

def min_table(ctx: CylonContext, table: Table, skipna=True) -> Column:
    cdef shared_ptr[CCylonContext] cctx = pycylon_unwrap_context(ctx)
    cdef shared_ptr[CTable] ctable = pycylon_unwrap_table(table)
    cdef CBasicOptions options = CBasicOptions(skipna)
    cdef shared_ptr[CColumn] cresult
    cdef CStatus status = MinTable(cctx, ctable, &cresult, options)

    if status.is_ok():
        return pycylon_wrap_column(cresult)
    else:
        raise Exception(f"aggregation error: {status.get_msg().decode()}")

def max_table(ctx: CylonContext, table: Table, skipna=True) -> Column:
    cdef shared_ptr[CCylonContext] cctx = pycylon_unwrap_context(ctx)
    cdef shared_ptr[CTable] ctable = pycylon_unwrap_table(table)
    cdef CBasicOptions options = CBasicOptions(skipna)
    cdef shared_ptr[CColumn] cresult
    cdef CStatus status = MaxTable(cctx, ctable, &cresult, options)

    if status.is_ok():
        return pycylon_wrap_column(cresult)
    else:
        raise Exception(f"aggregation error: {status.get_msg().decode()}")

# def count_table(ctx: CylonContext, table: Table) -> Column:
#     cdef shared_ptr[CCylonContext] cctx = pycylon_unwrap_context(ctx)
#     cdef shared_ptr[CTable] ctable = pycylon_unwrap_table(table)
#     cdef shared_ptr[CColumn] cresult
#     cdef CStatus status = CountTable(cctx, ctable, &cresult)
#
#     if status.is_ok():
#         return pycylon_wrap_column(cresult)
#     else:
#         raise Exception(f"aggregation error: {status.get_msg().decode()}")

def mean_table(ctx: CylonContext, table: Table, skipna=True) -> Column:
    cdef shared_ptr[CCylonContext] cctx = pycylon_unwrap_context(ctx)
    cdef shared_ptr[CTable] ctable = pycylon_unwrap_table(table)
    cdef CBasicOptions options = CBasicOptions(skipna)
    cdef shared_ptr[CColumn] cresult
    cdef CStatus status = MeanTable(cctx, ctable, &cresult, options)

    if status.is_ok():
        return pycylon_wrap_column(cresult)
    else:
        raise Exception(f"aggregation error: {status.get_msg().decode()}")

def var_table(ctx: CylonContext, table: Table, skipna: bool = True, ddof: int = 1) -> Column:
    cdef shared_ptr[CCylonContext] cctx = pycylon_unwrap_context(ctx)
    cdef shared_ptr[CTable] ctable = pycylon_unwrap_table(table)
    cdef CVarKernelOptions options = CVarKernelOptions(ddof, skipna)
    cdef shared_ptr[CColumn] cresult
    cdef CStatus status = VarTable(cctx, ctable, &cresult, options)

    if status.is_ok():
        return pycylon_wrap_column(cresult)
    else:
        raise Exception(f"aggregation error: {status.get_msg().decode()}")

def std_table(ctx: CylonContext, table: Table, skipna: bool = True, ddof: int = 1) -> Column:
    cdef shared_ptr[CCylonContext] cctx = pycylon_unwrap_context(ctx)
    cdef shared_ptr[CTable] ctable = pycylon_unwrap_table(table)
    cdef CVarKernelOptions options = CVarKernelOptions(ddof, skipna)
    cdef shared_ptr[CColumn] cresult
    cdef CStatus status = StdTable(cctx, ctable, &cresult, options)

    if status.is_ok():
        return pycylon_wrap_column(cresult)
    else:
        raise Exception(f"aggregation error: {status.get_msg().decode()}")
