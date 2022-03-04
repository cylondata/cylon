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


from pycylon.common.status cimport CStatus
from pycylon.data.table cimport CTable
from pycylon.data.scalar cimport CScalar
from pycylon.data.column cimport CColumn
from pycylon.ctx.context cimport CCylonContext
from libcpp.memory cimport shared_ptr
from libcpp cimport bool as c_bool

cdef extern from "../../../../cpp/src/cylon/compute/aggregate_kernels.hpp" namespace "cylon":
    cdef enum CGroupByAggregationOp 'cylon::compute::AggregationOpId':
        CSUM 'cylon::compute::SUM'
        CCOUNT 'cylon::compute::COUNT'
        CMIN 'cylon::compute::MIN'
        CMAX 'cylon::compute::MAX'
        CMEAN 'cylon::compute::MEAN'
        CVAR 'cylon::compute::VAR'
        CNUNIQUE 'cylon::compute::NUNIQUE'
        CQUANTILE 'cylon::compute::QUANTILE'
        CSTDDEV 'cylon::compute::STDDEV'

    cdef cppclass CBasicOptions "cylon::compute::BasicOptions":
        CBasicOptions()
        CBasicOptions(c_bool skip_nulls)

    cdef cppclass CVarKernelOptions "cylon::compute::VarKernelOptions":
        CVarKernelOptions()
        CVarKernelOptions(int ddof, c_bool skip_nulls)


cdef extern from "../../../../cpp/src/cylon/compute/aggregates.hpp" namespace "cylon::compute":
    CStatus Sum(const shared_ptr[CTable] & table, int col_idx, shared_ptr[CTable] & output)

    CStatus Count(const shared_ptr[CTable] & table, int col_idx, shared_ptr[CTable] & output)

    CStatus Min(const shared_ptr[CTable] & table, int col_idx, shared_ptr[CTable] & output)

    CStatus Max(const shared_ptr[CTable] & table, int col_idx, shared_ptr[CTable] & output)

    # Column, Scalar based API
    # cython does not support overloaded methods in cppclass. https://stackoverflow.com/a/42627030/4116268
    CStatus SumColumn "Sum"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CColumn] & values,
                             shared_ptr[CScalar] *result, const CBasicOptions & options)
    CStatus SumTable "Sum"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CTable] & table,
                            shared_ptr[CColumn] *result, const CBasicOptions & options)

    CStatus MinColumn "Min"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CColumn] & values,
                             shared_ptr[CScalar] *result, const CBasicOptions & options)
    CStatus MinTable "Min"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CTable] & table,
                            shared_ptr[CColumn] *result, const CBasicOptions & options)

    CStatus MaxColumn "Max"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CColumn] & values,
                             shared_ptr[CScalar] *result, const CBasicOptions & options)
    CStatus MaxTable "Max"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CTable] & table,
                            shared_ptr[CColumn] *result, const CBasicOptions & options)

    CStatus CountColumn "Count"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CColumn] & values,
                                 shared_ptr[CScalar] *result)
    CStatus CountTable "Count"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CTable] & table,
                                shared_ptr[CColumn] *result)

    CStatus MeanColumn "Mean"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CColumn] & values,
                               shared_ptr[CScalar] *result, const CBasicOptions & options)
    CStatus MeanTable "Mean"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CTable] & table,
                              shared_ptr[CColumn] *result, const CBasicOptions & options)

    CStatus VarColumn "Variance"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CColumn] & values,
                                  shared_ptr[CScalar] *result, const CVarKernelOptions & options)
    CStatus VarTable "Variance"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CTable] & table,
                                 shared_ptr[CColumn] *result, const CVarKernelOptions & options)

    CStatus StdColumn "StdDev"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CColumn] & values,
                                shared_ptr[CScalar] *result, const CVarKernelOptions & options)
    CStatus StdTable "StdDev"(const shared_ptr[CCylonContext] & ctx, const shared_ptr[CTable] & table,
                               shared_ptr[CColumn] *result, const CVarKernelOptions & options)
