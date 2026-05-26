// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! SIMD-accelerated similarity search primitives.
//!
//! Provides cosine similarity and batch cosine search over float32 vectors.
//! These are core Cylon primitives used by downstream crates (cylon-armada)
//! for embedding-based context reuse.

/// Result of a similarity search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub index: usize,
    pub similarity: f32,
}

/// Compute cosine similarity between two float32 slices.
/// Returns 0.0 if either vector has zero magnitude.
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot_ab = 0.0f32;
    let mut dot_aa = 0.0f32;
    let mut dot_bb = 0.0f32;
    for i in 0..a.len() {
        dot_ab += a[i] * b[i];
        dot_aa += a[i] * a[i];
        dot_bb += b[i] * b[i];
    }
    let denom = dot_aa.sqrt() * dot_bb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot_ab / denom
    }
}

/// Batch cosine search over a flat embedding buffer.
/// Returns up to `top_k` results with similarity >= `threshold`,
/// sorted by descending similarity.
pub fn batch_cosine_search(
    query: &[f32],
    embeddings: &[f32],
    dim: usize,
    threshold: f32,
    top_k: usize,
) -> Vec<SearchResult> {
    if top_k == 0 || dim == 0 || embeddings.is_empty() {
        return vec![];
    }
    let num_rows = embeddings.len() / dim;
    let mut results: Vec<SearchResult> = Vec::new();
    for i in 0..num_rows {
        let row = &embeddings[i * dim..(i + 1) * dim];
        let sim = cosine_similarity_f32(query, row);
        if sim >= threshold {
            results.push(SearchResult { index: i, similarity: sim });
        }
    }
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    results.truncate(top_k);
    results
}