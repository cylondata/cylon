/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CYLON_SIMD_OPS_HPP
#define CYLON_SIMD_OPS_HPP

#include <cstdint>
#include <vector>
#include <memory>

#include <arrow/array.h>

namespace cylon {
namespace simd {

/// Result of a similarity search: row index and cosine similarity score.
struct SearchResult {
  int64_t index;
  float similarity;
};

/// Compute cosine similarity between two float32 vectors of length @p dim.
/// Returns 0.0 if either vector has zero magnitude.
float cosine_similarity_f32(const float* a, const float* b, int dim);

/// Batch cosine search: compare @p query against @p num_rows embeddings
/// stored in a contiguous flat buffer (row-major: num_rows * dim floats).
/// Returns up to @p top_k results with similarity >= @p threshold,
/// sorted by descending similarity.
std::vector<SearchResult> batch_cosine_search(
    const float* query, int dim,
    const float* embeddings, int64_t num_rows,
    float threshold, int top_k);

/// Arrow-native batch cosine search: operates directly on a
/// FixedSizeList<Float32> column. Zero-copy — reads the underlying
/// contiguous values buffer without copying.
std::vector<SearchResult> batch_cosine_search_arrow(
    const float* query, int dim,
    const std::shared_ptr<arrow::FixedSizeListArray>& embeddings,
    float threshold, int top_k);

}  // namespace simd
}  // namespace cylon

#endif  // CYLON_SIMD_OPS_HPP