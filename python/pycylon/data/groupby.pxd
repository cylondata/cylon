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
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from pycylon.data.aggregates cimport CGroupByAggregationOp


cdef extern from "../../../cpp/src/cylon/groupby/groupby.hpp" namespace "cylon":

    CStatus GroupBy(shared_ptr[CTable] &table, int index_col, const vector[long]
                     &aggregate_cols, const vector[CGroupByAggregationOp] &aggregate_ops,
                     shared_ptr[CTable] &output)

    CStatus PipelineGroupBy(shared_ptr[CTable] &table, int index_col, const vector[long]
                             &aggregate_cols, const vector[CGroupByAggregationOp] &aggregate_ops,
                             shared_ptr[CTable] &output)
