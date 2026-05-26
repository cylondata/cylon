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

//! SIMD-optimized operations for WASM
//!
//! This module provides SIMD128 optimized implementations for common
//! aggregation and computation operations. WASM SIMD provides 128-bit
//! vectors, allowing 4-wide f32 operations or 2-wide f64 operations.
//!
//! See docs/SIMD_VECTORIZATION.md for detailed explanation.

use wasm_bindgen::prelude::*;

#[cfg(feature = "simd")]
use std::arch::wasm32::*;

// =============================================================================
// Sum Operations
// =============================================================================

/// SIMD-optimized sum for f32 arrays (4-wide vectorization)
#[cfg(feature = "simd")]
pub fn simd_sum_f32(data: &[f32]) -> f32 {
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut acc = unsafe { f32x4_splat(0.0) };

    for chunk in chunks {
        let v = unsafe { f32x4(chunk[0], chunk[1], chunk[2], chunk[3]) };
        acc = unsafe { f32x4_add(acc, v) };
    }

    let sum = unsafe {
        f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc)
    };

    sum + remainder.iter().sum::<f32>()
}

#[cfg(not(feature = "simd"))]
pub fn simd_sum_f32(data: &[f32]) -> f32 {
    data.iter().sum()
}

/// SIMD-optimized sum for f64 arrays (2-wide vectorization)
#[cfg(feature = "simd")]
pub fn simd_sum_f64(data: &[f64]) -> f64 {
    let chunks = data.chunks_exact(2);
    let remainder = chunks.remainder();

    let mut acc = unsafe { f64x2_splat(0.0) };

    for chunk in chunks {
        let v = unsafe { f64x2(chunk[0], chunk[1]) };
        acc = unsafe { f64x2_add(acc, v) };
    }

    let sum = unsafe {
        f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc)
    };

    sum + remainder.iter().sum::<f64>()
}

#[cfg(not(feature = "simd"))]
pub fn simd_sum_f64(data: &[f64]) -> f64 {
    data.iter().sum()
}

// =============================================================================
// Dot Product Operations
// =============================================================================

/// SIMD-optimized dot product for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Arrays must have equal length");

    let a_chunks = a.chunks_exact(4);
    let b_chunks = b.chunks_exact(4);
    let a_remainder = a_chunks.remainder();
    let b_remainder = b_chunks.remainder();

    let mut acc = unsafe { f32x4_splat(0.0) };

    for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
        let va = unsafe { f32x4(a_chunk[0], a_chunk[1], a_chunk[2], a_chunk[3]) };
        let vb = unsafe { f32x4(b_chunk[0], b_chunk[1], b_chunk[2], b_chunk[3]) };
        let product = unsafe { f32x4_mul(va, vb) };
        acc = unsafe { f32x4_add(acc, product) };
    }

    let sum = unsafe {
        f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc)
    };

    let remainder_sum: f32 = a_remainder
        .iter()
        .zip(b_remainder.iter())
        .map(|(a, b)| a * b)
        .sum();

    sum + remainder_sum
}

#[cfg(not(feature = "simd"))]
pub fn simd_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Arrays must have equal length");
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

/// SIMD-optimized dot product for f64 arrays
#[cfg(feature = "simd")]
pub fn simd_dot_product_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Arrays must have equal length");

    let a_chunks = a.chunks_exact(2);
    let b_chunks = b.chunks_exact(2);
    let a_remainder = a_chunks.remainder();
    let b_remainder = b_chunks.remainder();

    let mut acc = unsafe { f64x2_splat(0.0) };

    for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
        let va = unsafe { f64x2(a_chunk[0], a_chunk[1]) };
        let vb = unsafe { f64x2(b_chunk[0], b_chunk[1]) };
        let product = unsafe { f64x2_mul(va, vb) };
        acc = unsafe { f64x2_add(acc, product) };
    }

    let sum = unsafe {
        f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc)
    };

    let remainder_sum: f64 = a_remainder
        .iter()
        .zip(b_remainder.iter())
        .map(|(a, b)| a * b)
        .sum();

    sum + remainder_sum
}

#[cfg(not(feature = "simd"))]
pub fn simd_dot_product_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Arrays must have equal length");
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

// =============================================================================
// Min/Max Operations
// =============================================================================

/// SIMD-optimized min for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_min_f32(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }

    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut acc = unsafe { f32x4_splat(f32::INFINITY) };

    for chunk in chunks {
        let v = unsafe { f32x4(chunk[0], chunk[1], chunk[2], chunk[3]) };
        acc = unsafe { f32x4_min(acc, v) };
    }

    let mut min = unsafe {
        f32x4_extract_lane::<0>(acc)
            .min(f32x4_extract_lane::<1>(acc))
            .min(f32x4_extract_lane::<2>(acc))
            .min(f32x4_extract_lane::<3>(acc))
    };

    for &val in remainder {
        min = min.min(val);
    }

    Some(min)
}

#[cfg(not(feature = "simd"))]
pub fn simd_min_f32(data: &[f32]) -> Option<f32> {
    data.iter().copied().reduce(f32::min)
}

/// SIMD-optimized max for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_max_f32(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }

    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut acc = unsafe { f32x4_splat(f32::NEG_INFINITY) };

    for chunk in chunks {
        let v = unsafe { f32x4(chunk[0], chunk[1], chunk[2], chunk[3]) };
        acc = unsafe { f32x4_max(acc, v) };
    }

    let mut max = unsafe {
        f32x4_extract_lane::<0>(acc)
            .max(f32x4_extract_lane::<1>(acc))
            .max(f32x4_extract_lane::<2>(acc))
            .max(f32x4_extract_lane::<3>(acc))
    };

    for &val in remainder {
        max = max.max(val);
    }

    Some(max)
}

#[cfg(not(feature = "simd"))]
pub fn simd_max_f32(data: &[f32]) -> Option<f32> {
    data.iter().copied().reduce(f32::max)
}

// =============================================================================
// Similarity/Distance Operations
// =============================================================================

/// SIMD-optimized cosine similarity for f32 arrays
/// cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)
pub fn simd_cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot = simd_dot_product_f32(a, b);
    let norm_a = simd_dot_product_f32(a, a).sqrt();
    let norm_b = simd_dot_product_f32(b, b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// SIMD-optimized euclidean distance for f32 arrays
/// ||a - b|| = sqrt(sum((a[i] - b[i])^2))
#[cfg(feature = "simd")]
pub fn simd_euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Arrays must have equal length");

    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    let mut acc = unsafe { f32x4_splat(0.0) };

    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
        let va = unsafe { f32x4(chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3]) };
        let vb = unsafe { f32x4(chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3]) };
        let diff = unsafe { f32x4_sub(va, vb) };
        let sq = unsafe { f32x4_mul(diff, diff) };
        acc = unsafe { f32x4_add(acc, sq) };
    }

    let sum = unsafe {
        f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc)
    };

    let remainder_sum: f32 = remainder_a
        .iter()
        .zip(remainder_b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    (sum + remainder_sum).sqrt()
}

#[cfg(not(feature = "simd"))]
pub fn simd_euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Arrays must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

// =============================================================================
// WASM Exports
// =============================================================================

/// Compute sum of f32 array
#[wasm_bindgen]
pub fn sum_f32(data: &[f32]) -> f32 {
    simd_sum_f32(data)
}

/// Compute sum of f64 array
#[wasm_bindgen]
pub fn sum_f64(data: &[f64]) -> f64 {
    simd_sum_f64(data)
}

/// Compute dot product of two f32 arrays
#[wasm_bindgen]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    simd_dot_product_f32(a, b)
}

/// Compute cosine similarity of two f32 arrays
#[wasm_bindgen]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    simd_cosine_similarity_f32(a, b)
}

/// Compute euclidean distance between two f32 arrays
#[wasm_bindgen]
pub fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    simd_euclidean_distance_f32(a, b)
}

/// Batch cosine search: compare query against all embeddings in a flat buffer.
/// Returns JSON string array of {index, similarity} sorted by descending similarity.
///
/// # Arguments
/// * `query` - Query vector (f32)
/// * `embeddings` - Flat buffer of embeddings (num_rows * dim floats)
/// * `dim` - Embedding dimension
/// * `threshold` - Minimum cosine similarity
/// * `top_k` - Maximum results to return
#[wasm_bindgen]
pub fn batch_cosine_search_f32(
    query: &[f32],
    embeddings: &[f32],
    dim: usize,
    threshold: f32,
    top_k: usize,
) -> Result<String, JsValue> {
    let results = cylon::simd::batch_cosine_search(query, embeddings, dim, threshold, top_k);
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| serde_json::json!({"index": r.index, "similarity": r.similarity}))
        .collect();
    serde_json::to_string(&json_results)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
