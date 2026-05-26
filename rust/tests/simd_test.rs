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

use cylon::simd::{batch_cosine_search, cosine_similarity_f32};

#[test]
fn test_cosine_identical() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    assert!((cosine_similarity_f32(&a, &a) - 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_orthogonal() {
    let a = vec![1.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];
    assert!(cosine_similarity_f32(&a, &b).abs() < 1e-5);
}

#[test]
fn test_cosine_opposite() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![-1.0, -2.0, -3.0];
    assert!((cosine_similarity_f32(&a, &b) + 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_zero_vector() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![0.0, 0.0, 0.0];
    assert_eq!(cosine_similarity_f32(&a, &b), 0.0);
}

#[test]
fn test_batch_search_basic() {
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let embeddings = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.9, 0.1, 0.0, 0.0,
    ];
    let results = batch_cosine_search(&query, &embeddings, 4, 0.5, 10);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].index, 0);
    assert_eq!(results[1].index, 2);
}

#[test]
fn test_batch_search_top_k() {
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let embeddings = vec![
        1.0, 0.0, 0.0, 0.0,
        0.9, 0.1, 0.0, 0.0,
        0.8, 0.2, 0.0, 0.0,
    ];
    let results = batch_cosine_search(&query, &embeddings, 4, 0.0, 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].index, 0);
}

#[test]
fn test_batch_search_top_k_zero() {
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let embeddings = vec![1.0, 0.0, 0.0, 0.0];
    let results = batch_cosine_search(&query, &embeddings, 4, 0.0, 0);
    assert!(results.is_empty());
}