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

#include <glog/logging.h>
#include <cylon/join/hash_join.hpp>
#include <cylon/arrow/arrow_comparator.hpp>

namespace cylon {
namespace join {

template<typename HASHMAP_T>
inline void probe_hash_map_no_fill(const HASHMAP_T &hash_map,
                                   const int64_t probe_tab_len,
                                   std::vector<int64_t> &build_tab_indices,
                                   std::vector<int64_t> &probe_tab_indices) {
  for (int64_t i = 0; i < probe_tab_len; i++) {
    const auto &res = hash_map.equal_range(cylon::util::SetBit(i));
    for (auto it = res.first; it != res.second; it++) {
      build_tab_indices.push_back(it->second); // no need to unset bit here bcs
      probe_tab_indices.push_back(i);
    }
  }
}

template<typename HASHMAP_T>
inline void probe_hash_map_with_fill(const HASHMAP_T &hash_map,
                                     const int64_t probe_tab_len,
                                     std::vector<int64_t> &build_tab_indices,
                                     std::vector<int64_t> &probe_tab_indices) {
  for (int64_t i = 0; i < probe_tab_len; i++) {
    const auto &res = hash_map.equal_range(cylon::util::SetBit(i));
    if (res.first == res.second) { // no matching rows from hashmap
      build_tab_indices.push_back(-1);
      probe_tab_indices.push_back(i);
    } else {
      for (auto it = res.first; it != res.second; it++) {
        build_tab_indices.push_back(it->second);
        probe_tab_indices.push_back(i);
      }
    }
  }
}

template<typename HASHMAP_T>
inline void probe_hash_map_outer(const HASHMAP_T &hash_map,
                                 const int64_t build_tab_len,
                                 const int64_t probe_tab_len,
                                 std::vector<int64_t> &build_tab_indices,
                                 std::vector<int64_t> &probe_tab_indices) {
  std::vector<bool> build_tab_survivors(build_tab_len, true),
      probe_tab_survivors(probe_tab_len, true);

  for (int64_t i = 0; i < probe_tab_len; ++i) {
    const auto &range = hash_map.equal_range(cylon::util::SetBit(i));
    // if found, range.first != range.second. so probe_tab_survivors[i] = false
    probe_tab_survivors[i] = (range.first == range.second);
    for (auto it = range.first; it != range.second; it++) {
      build_tab_indices.push_back(it->second);
      build_tab_survivors[it->second] = false; // unset build table inidces

      probe_tab_indices.push_back(i);
    }
  }

  for (size_t i = 0; i < build_tab_survivors.size(); i++) {
    if (build_tab_survivors[i]) {
      build_tab_indices.push_back((int64_t) i);
      probe_tab_indices.push_back(-1);
    }
  }

  for (size_t i = 0; i < probe_tab_survivors.size(); i++) {
    if (probe_tab_survivors[i]) {
      build_tab_indices.push_back(-1);
      probe_tab_indices.push_back((int64_t) i);
    }
  }
}

template<typename HASHMAP_T>
inline void do_probe(cylon::join::config::JoinType join_type,
                     const HASHMAP_T &hash_map,
                     int64_t build_size,
                     int64_t probe_size,
                     std::vector<int64_t> &build_tab_indices,
                     std::vector<int64_t> &probe_tab_indices) {
  switch (join_type) {
    case config::LEFT:
    case config::RIGHT: {
      probe_hash_map_with_fill(hash_map, probe_size, build_tab_indices, probe_tab_indices);
      break;
    }
    case config::INNER: {
      probe_hash_map_no_fill(hash_map, probe_size, build_tab_indices, probe_tab_indices);
      break;
    }
    case config::FULL_OUTER: {
      probe_hash_map_outer(hash_map, build_size, probe_size, build_tab_indices,
                           probe_tab_indices);
      break;
    }
  }
}

inline void calculate_metadata(cylon::join::config::JoinType join_type, int64_t left_size,
                               int64_t right_size, bool *build_idx, int64_t *init_vec_size) {
  switch (join_type) {
    case config::LEFT: {
      *build_idx = true; // i.e. build hash map from right table, probe from left
      *init_vec_size = left_size;
      break;
    }
    case config::RIGHT: {
      *build_idx = false; // i.e. build hash map from left table, probe from right
      *init_vec_size = right_size;
      break;
    }
    case config::INNER: {
      *init_vec_size = std::min(left_size, right_size);
      goto init_ptrs;
    }
    case config::FULL_OUTER: {
      *init_vec_size = left_size + right_size;
      init_ptrs:
      if (left_size <= right_size) {
        *build_idx = false; // i.e. build hash map from left table, probe from right
      } else {
        *build_idx = true; // i.e. build hash map from right table, probe from left
      }
      break;
    }
  }
}

Status multi_index_hash_join(const std::shared_ptr<arrow::Table> &ltab,
                             const std::shared_ptr<arrow::Table> &rtab,
                             const config::JoinConfig &config,
                             std::shared_ptr<arrow::Table> *joined_table,
                             arrow::MemoryPool *memory_pool) {
  if (ltab->column(0)->num_chunks() > 1 || rtab->column(0)->num_chunks() > 1) {
    return {Code::Invalid, "left or right table has chunked arrays"};
  }

  // 2 element arrays containing [left, right] info
  const std::array<const std::shared_ptr<arrow::Table> *, 2> tabs{&ltab, &rtab};
  const std::array<const std::vector<int> *, 2>
      col_indices{&config.GetLeftColumnIdx(), &config.GetRightColumnIdx()};
  std::array<std::vector<int64_t>, 2> row_indices{std::vector<int64_t>{}, std::vector<int64_t>{}};

  using TwoTableRowIndexHashMMap = typename std::unordered_multimap<int64_t, int64_t,
                                                                    DualTableRowIndexHash,
                                                                    DualTableRowIndexEqualTo>;

  int64_t init_vec_size = 0;
  // if 0: build from left and probe from right; else: build from right and probe from left
  bool build_idx = false;

  // set up
  calculate_metadata(config.GetType(), ltab->num_rows(), rtab->num_rows(), &build_idx,
                     &init_vec_size);

  // reserve space for index vectors
  row_indices[0].reserve(init_vec_size);
  row_indices[1].reserve(init_vec_size);

  const int64_t build_size = (*tabs[build_idx])->num_rows();
  const int64_t probe_size = (*tabs[!build_idx])->num_rows();

  // populate left hash table
  std::unique_ptr<DualTableRowIndexHash> hash;
  RETURN_CYLON_STATUS_IF_FAILED (DualTableRowIndexHash::Make(*tabs[build_idx],
                                                             *tabs[!build_idx],
                                                             *col_indices[build_idx],
                                                             *col_indices[!build_idx],
                                                             &hash));

  std::unique_ptr<DualTableRowIndexEqualTo> equal_to;
  RETURN_CYLON_STATUS_IF_FAILED(DualTableRowIndexEqualTo::Make(*tabs[build_idx],
                                                               *tabs[!build_idx],
                                                               *col_indices[build_idx],
                                                               *col_indices[!build_idx],
                                                               &equal_to));

  TwoTableRowIndexHashMMap hash_map((*tabs[build_idx])->num_rows(), *hash, *equal_to);

  // build hashmap from corresponding table
  for (int64_t i = 0; i < build_size; i++) {
    hash_map.emplace(i, i);
  }

  // probe
  do_probe(config.GetType(), hash_map, build_size, probe_size, row_indices[build_idx],
           row_indices[!build_idx]);

  // clean up
  hash_map.clear();

  // copy arrays from the table indices
  return util::build_final_table(row_indices[0], row_indices[1],
                                 ltab, rtab, config.GetLeftTableSuffix(),
                                 config.GetRightTableSuffix(), joined_table, memory_pool);
}

Status ArrayIndexHashJoin(const std::shared_ptr<arrow::Array> &left_idx_col,
                          const std::shared_ptr<arrow::Array> &right_idx_col,
                          config::JoinType join_type,
                          std::vector<int64_t> &left_table_indices,
                          std::vector<int64_t> &right_table_indices) {
  if (left_idx_col->type_id() != right_idx_col->type_id()) {
    return {Code::Invalid, "left and right index array types are not equal"};
  }

  // arrays for house-keeping
  const std::array<const std::shared_ptr<arrow::Array> *, 2> arrays{&left_idx_col, &right_idx_col};
  const std::array<std::vector<int64_t> *, 2>
      row_indices{&left_table_indices, &right_table_indices};

  // number of elems to be used to initialize indices vectors
  int64_t init_vec_size = 0;
  // if 0: build from left and probe from right; else: build from right and probe from left
  bool build_idx = false;

  // setup
  calculate_metadata(join_type, left_idx_col->length(), right_idx_col->length(), &build_idx,
                     &init_vec_size);

  // reserve space for index vectors
  row_indices[0]->reserve(init_vec_size);
  row_indices[1]->reserve(init_vec_size);

  const int64_t build_size = (*arrays[build_idx])->length();
  const int64_t probe_size = (*arrays[!build_idx])->length();

  // precalculate hashes
  std::unique_ptr<HashPartitionKernel> hash_kernel;
  RETURN_CYLON_STATUS_IF_FAILED(CreateHashPartitionKernel(left_idx_col->type(), &hash_kernel));
  std::array<std::vector<uint32_t>, 2>
      hashes{std::vector<uint32_t>(build_size, 0),
             std::vector<uint32_t>(probe_size, 0)};
  hash_kernel->UpdateHash(*arrays[build_idx], hashes[0]); // calc hashes of left array
  hash_kernel->UpdateHash(*arrays[!build_idx], hashes[1]); // calc hashes of right array

  const auto &hash = [&hashes](const int64_t idx) -> uint32_t { // hash lambda
    return hashes[cylon::util::CheckBit(idx)][cylon::util::ClearBit(idx)];
  };

  // comparator
  std::unique_ptr<DualArrayIndexComparator> comp;
  RETURN_CYLON_STATUS_IF_FAILED(CreateDualArrayIndexComparator(*arrays[build_idx],
                                                               *arrays[!build_idx],
                                                               &comp));
  const auto
      &equal_to = [&comp](const int64_t idx1, const int64_t idx2) -> bool { // equal_to lambda
    return comp->equal_to(idx1, idx2);
  };

  // define mutimap type for convenience
  using TwoArrayIndexHashMMap = typename std::unordered_multimap<int64_t, int64_t,
                                                                 decltype(hash),
                                                                 decltype(equal_to)>;

  // populate left hash table. use build array length number of buckets in the hashmap
  TwoArrayIndexHashMMap hash_map((size_t) (*arrays[build_idx])->length(), hash, equal_to);

  // build hashmap from corresponding table
  for (int64_t i = 0; i < build_size; i++) {
    hash_map.emplace(i, i);
  }
  hashes[0].clear(); // hashes of the building array are no longer needed
  // probe

  do_probe(join_type, hash_map, build_size, probe_size, *row_indices[build_idx],
           *row_indices[!build_idx]);
  hashes[1].clear();

  return Status::OK();
}

Status HashJoin(const std::shared_ptr<arrow::Table> &ltab,
                const std::shared_ptr<arrow::Table> &rtab,
                const config::JoinConfig &config,
                std::shared_ptr<arrow::Table> *joined_table,
                arrow::MemoryPool *memory_pool) {
  if (config.GetLeftColumnIdx().size() != config.GetRightColumnIdx().size()) {
    return {Code::Invalid, "left and right index vector sizes should be the same"};
  }

  // let's first combine chunks in both tables
  std::shared_ptr<arrow::Table> c_ltab(ltab);
  COMBINE_CHUNKS_RETURN_CYLON_STATUS(c_ltab, memory_pool);

  std::shared_ptr<arrow::Table> c_rtab(rtab);
  COMBINE_CHUNKS_RETURN_CYLON_STATUS(c_rtab, memory_pool);

  if (config.GetLeftColumnIdx().size() == 1) {
    int left_idx = config.GetLeftColumnIdx()[0];
    int right_idx = config.GetRightColumnIdx()[0];

    std::vector<int64_t> left_indices, right_indices;

    RETURN_CYLON_STATUS_IF_FAILED(
        ArrayIndexHashJoin(cylon::util::GetChunkOrEmptyArray(c_ltab->column(left_idx), 0),
                           cylon::util::GetChunkOrEmptyArray(c_rtab->column(right_idx), 0),
                           config.GetType(),
                           left_indices,
                           right_indices));

    return util::build_final_table(left_indices, right_indices, c_ltab, c_rtab,
                                   config.GetLeftTableSuffix(), config.GetRightTableSuffix(),
                                   joined_table, memory_pool);
  } else {
    return multi_index_hash_join(c_ltab, c_rtab, config, joined_table, memory_pool);
  }
}

}
}