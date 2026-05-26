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

#include "simd_ops.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <utility>

#if defined(CYLON_HAVE_AVX2)
#include <immintrin.h>
#elif defined(CYLON_HAVE_SSE4_2)
#include <smmintrin.h>
#elif defined(CYLON_HAVE_NEON)
#include <arm_neon.h>
#endif

namespace cylon {
namespace simd {

// ---------------------------------------------------------------------------
// ISA-specific dot product kernels
// ---------------------------------------------------------------------------

#if defined(CYLON_HAVE_AVX2)

static float dot_product_f32(const float* a, const float* b, int dim) {
  __m256 sum = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum = _mm256_fmadd_ps(va, vb, sum);
  }
  // Horizontal reduction: 8 → 1
  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float result = _mm_cvtss_f32(s);
  // Scalar tail
  for (; i < dim; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

#elif defined(CYLON_HAVE_SSE4_2)

static float dot_product_f32(const float* a, const float* b, int dim) {
  __m128 sum = _mm_setzero_ps();
  int i = 0;
  for (; i + 4 <= dim; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
  }
  // Horizontal reduction
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float result = _mm_cvtss_f32(sum);
  for (; i < dim; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

#elif defined(CYLON_HAVE_NEON)

static float dot_product_f32(const float* a, const float* b, int dim) {
  float32x4_t sum = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 4 <= dim; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    sum = vfmaq_f32(sum, va, vb);
  }
  float result = vaddvq_f32(sum);
  for (; i < dim; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

#else  // Scalar fallback

static float dot_product_f32(const float* a, const float* b, int dim) {
  float result = 0.0f;
  for (int i = 0; i < dim; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

#endif

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

float cosine_similarity_f32(const float* a, const float* b, int dim) {
  float dot_ab = dot_product_f32(a, b, dim);
  float dot_aa = dot_product_f32(a, a, dim);
  float dot_bb = dot_product_f32(b, b, dim);
  float denom = std::sqrt(dot_aa) * std::sqrt(dot_bb);
  if (denom == 0.0f) {
    return 0.0f;
  }
  return dot_ab / denom;
}

std::vector<SearchResult> batch_cosine_search(
    const float* query, int dim,
    const float* embeddings, int64_t num_rows,
    float threshold, int top_k) {
  if (top_k <= 0 || num_rows <= 0 || dim <= 0) {
    return {};
  }

  // Min-heap of (similarity, index) — keeps the top-k highest similarities
  using Pair = std::pair<float, int64_t>;
  std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;

  // Precompute query norm
  float query_norm_sq = dot_product_f32(query, query, dim);
  if (query_norm_sq == 0.0f) {
    return {};
  }
  float query_norm = std::sqrt(query_norm_sq);

  for (int64_t i = 0; i < num_rows; ++i) {
    const float* row = embeddings + i * dim;
    float dot_qr = dot_product_f32(query, row, dim);
    float row_norm_sq = dot_product_f32(row, row, dim);
    if (row_norm_sq == 0.0f) {
      continue;
    }
    float sim = dot_qr / (query_norm * std::sqrt(row_norm_sq));
    if (sim >= threshold) {
      if (static_cast<int>(heap.size()) < top_k) {
        heap.emplace(sim, i);
      } else if (sim > heap.top().first) {
        heap.pop();
        heap.emplace(sim, i);
      }
    }
  }

  // Extract results sorted by descending similarity
  std::vector<SearchResult> results;
  results.reserve(heap.size());
  while (!heap.empty()) {
    auto [sim, idx] = heap.top();
    heap.pop();
    results.push_back({idx, sim});
  }
  std::reverse(results.begin(), results.end());
  return results;
}

std::vector<SearchResult> batch_cosine_search_arrow(
    const float* query, int dim,
    const std::shared_ptr<arrow::FixedSizeListArray>& embeddings,
    float threshold, int top_k) {
  // FixedSizeList<Float32> stores all values in a single contiguous Float32Array
  auto values = std::static_pointer_cast<arrow::FloatArray>(embeddings->values());
  const float* data = values->raw_values();
  int64_t num_rows = embeddings->length();
  return batch_cosine_search(query, dim, data, num_rows, threshold, top_k);
}

}  // namespace simd
}  // namespace cylon