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

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libc.stdint cimport int64_t

from pyarrow.lib cimport CArray


cdef extern from "../../../../cpp/src/cylon/simd/simd_ops.hpp" namespace "cylon::simd":
    cdef struct CSearchResult "cylon::simd::SearchResult":
        int64_t index
        float similarity

    float cosine_similarity_f32(const float* a, const float* b, int dim)

    vector[CSearchResult] batch_cosine_search(
        const float* query, int dim,
        const float* embeddings, int64_t num_rows,
        float threshold, int top_k)