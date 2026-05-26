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

import numpy as np
from libc.stdint cimport int64_t
from libcpp.vector cimport vector

from pycylon.simd.simd cimport (
    CSearchResult,
    cosine_similarity_f32 as c_cosine_similarity_f32,
    batch_cosine_search as c_batch_cosine_search,
)


def cosine_similarity(object a not None, object b not None):
    """Compute cosine similarity between two float32 vectors using SIMD.

    Args:
        a: First vector (numpy float32 array).
        b: Second vector (numpy float32 array, same length as a).

    Returns:
        Cosine similarity as a float in [-1, 1]. Returns 0.0 if either
        vector has zero magnitude.
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    cdef int dim = len(a)
    cdef const float* a_ptr = <const float*> (<unsigned long long> a.ctypes.data)
    cdef const float* b_ptr = <const float*> (<unsigned long long> b.ctypes.data)
    return c_cosine_similarity_f32(a_ptr, b_ptr, dim)


def batch_search(object query not None,
                 object embeddings not None,
                 float threshold=0.85,
                 int top_k=5):
    """Batch cosine search: compare query against all embeddings using SIMD.

    Args:
        query: Query vector (numpy float32 array, shape [dim]).
        embeddings: Embedding matrix (numpy float32 array, shape [num_rows, dim],
                    C-contiguous).
        threshold: Minimum cosine similarity to include in results.
        top_k: Maximum number of results to return.

    Returns:
        List of dicts with keys 'index' (int) and 'similarity' (float),
        sorted by descending similarity.
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if query.ndim != 1 or embeddings.ndim != 2:
        raise ValueError("query must be 1-D, embeddings must be 2-D")
    if len(query) != embeddings.shape[1]:
        raise ValueError(
            f"dimension mismatch: query dim {len(query)} "
            f"vs embedding dim {embeddings.shape[1]}")

    cdef int dim = len(query)
    cdef int64_t num_rows = embeddings.shape[0]
    cdef const float* q_ptr = <const float*> (<unsigned long long> query.ctypes.data)
    cdef const float* e_ptr = <const float*> (<unsigned long long> embeddings.ctypes.data)

    cdef vector[CSearchResult] results = c_batch_cosine_search(
        q_ptr, dim, e_ptr, num_rows,
        threshold, top_k)

    return [{"index": r.index, "similarity": r.similarity} for r in results]