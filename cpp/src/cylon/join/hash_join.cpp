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

#include "hash_join.hpp"
#include "arrow/arrow_comparator.hpp"

namespace cylon {
namespace join {

using TwoTableRowIndexHashMMap = typename std::unordered_multimap<int64_t,
                                                                  int64_t,
                                                                  TwoTableRowIndexHash,
                                                                  TwoTableRowIndexEqualTo>;

using TwoArrayIndexHashMMap = typename std::unordered_multimap<int64_t,
                                                               int64_t,
                                                               TwoArrayIndexHash,
                                                               TwoArrayIndexEqualTo>;

template<typename HASHMAP_T>
static void probe_hash_map_no_fill(const HASHMAP_T &hash_map,
                                   const int64_t build_tab_len,
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
static void probe_hash_map_with_fill(const HASHMAP_T &hash_map,
                                     const int64_t build_tab_len,
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
static void probe_hash_map_outer(const HASHMAP_T &hash_map,
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
      build_tab_indices.push_back(i);
      probe_tab_indices.push_back(-1);
    }
  }

  for (size_t i = 0; i < probe_tab_survivors.size(); i++) {
    if (probe_tab_survivors[i]) {
      build_tab_indices.push_back(-1);
      probe_tab_indices.push_back(i);
    }
  }
}

arrow::Status MultiIndexHashJoin(const std::shared_ptr<arrow::Table> &ltab,
                                 const std::shared_ptr<arrow::Table> &rtab,
                                 const config::JoinConfig &config,
                                 std::shared_ptr<arrow::Table> *joined_table,
                                 arrow::MemoryPool *memory_pool) {
  if (ltab->column(0)->num_chunks() > 1 || rtab->column(0)->num_chunks() > 1) {
    return arrow::Status::Invalid("left or right table has chunked arrays");
  }

  // 2 element arrays containing [left, right] info
  const std::array<const std::shared_ptr<arrow::Table> *, 2> tabs{&ltab, &rtab};
  const std::array<const std::vector<int> *, 2> col_indices{&config.GetLeftColumnIdx(), &config.GetRightColumnIdx()};
  std::array<std::vector<int64_t>, 2> row_indices{std::vector<int64_t>{}, std::vector<int64_t>{}};

  void (*probe_func_ptr)(const TwoTableRowIndexHashMMap &hash_map,
                         const int64_t build_tab_len,
                         const int64_t probe_tab_len,
                         std::vector<int64_t> &build_tab_indices,
                         std::vector<int64_t> &probe_tab_indices) = nullptr;
  size_t init_vec_size = 0;
  bool build_idx = false; // if 0: build from left and probe from right; else: build from right and probe from left

  switch (config.GetType()) {
    case config::LEFT: {
      build_idx = true; // i.e. build hash map from right table, probe from left
      init_vec_size = ltab->num_rows();

      probe_func_ptr = &probe_hash_map_with_fill<TwoTableRowIndexHashMMap>;
      break;
    }
    case config::RIGHT: {
      build_idx = false; // i.e. build hash map from left table, probe from right
      init_vec_size = rtab->num_rows();

      probe_func_ptr = &probe_hash_map_with_fill<TwoTableRowIndexHashMMap>;
      break;
    }
    case config::INNER: {
      init_vec_size = std::min(ltab->num_rows(), rtab->num_rows());
      probe_func_ptr = &probe_hash_map_no_fill<TwoTableRowIndexHashMMap>;
      goto init_ptrs;
    }
    case config::FULL_OUTER: {
      init_vec_size = ltab->num_rows() + rtab->num_rows();
      probe_func_ptr = &probe_hash_map_outer<TwoTableRowIndexHashMMap>;

      init_ptrs:
      if (ltab->num_rows() <= rtab->num_rows()) {
        build_idx = false; // i.e. build hash map from left table, probe from right
      } else {
        build_idx = true; // i.e. build hash map from right table, probe from left
      }
      break;
    }
  }

  // reserve space for index vectors
  row_indices[0].reserve(init_vec_size);
  row_indices[1].reserve(init_vec_size);

  // populate left hash table
  TwoTableRowIndexHash hash(*tabs[build_idx], *tabs[!build_idx], *col_indices[build_idx], *col_indices[!build_idx]);
  TwoTableRowIndexEqualTo
      equal_to(*tabs[build_idx], *tabs[!build_idx], *col_indices[build_idx], *col_indices[!build_idx]);

  TwoTableRowIndexHashMMap hash_map((*tabs[build_idx])->num_rows(), hash, equal_to);

  // build hashmap from corresponding table
  for (int64_t i = 0; i < (*tabs[build_idx])->num_rows(); i++) {
    hash_map.emplace(i, i);
  }

  // probe
  probe_func_ptr(hash_map, (*tabs[build_idx])->num_rows(), (*tabs[!build_idx])->num_rows(),
                 row_indices[build_idx], row_indices[!build_idx]);

  // clean up
  hash_map.clear();

  // copy arrays from the table indices
  // todo use arrow::compute::Take for this
  return cylon::join::util::build_final_table(row_indices[0],
                                              row_indices[1],
                                              ltab,
                                              rtab,
                                              config.GetLeftTableSuffix(),
                                              config.GetRightTableSuffix(),
                                              joined_table,
                                              memory_pool);
}

arrow::Status ArrayIndexHashJoin(const std::shared_ptr<arrow::Array> &left_idx_col,
                                 const std::shared_ptr<arrow::Array> &right_idx_col,
                                 config::JoinType join_type,
                                 std::vector<int64_t> &left_table_indices,
                                 std::vector<int64_t> &right_table_indices) {

  // let's first combine chunks in both tables
  const std::array<const std::shared_ptr<arrow::Array> *, 2> arrays{&left_idx_col, &right_idx_col};
  const std::array<std::vector<int64_t> *, 2> row_indices{&left_table_indices, &right_table_indices};

  void (*probe_func_ptr)(const TwoArrayIndexHashMMap &hash_map,
                         const int64_t build_tab_len,
                         const int64_t probe_tab_len,
                         std::vector<int64_t> &build_tab_indices,
                         std::vector<int64_t> &probe_tab_indices) = nullptr;
  size_t init_vec_size = 0;
  bool build_idx = false; // if 0: build from left and probe from right; else: build from right and probe from left

  switch (join_type) {
    case config::LEFT: {
      build_idx = true; // i.e. build hash map from right table, probe from left
      init_vec_size = left_idx_col->length();

      probe_func_ptr = &probe_hash_map_with_fill<TwoArrayIndexHashMMap>;
      break;
    }
    case config::RIGHT: {
      build_idx = false; // i.e. build hash map from left table, probe from right
      init_vec_size = right_idx_col->length();

      probe_func_ptr = &probe_hash_map_with_fill<TwoArrayIndexHashMMap>;
      break;
    }
    case config::INNER: {
      init_vec_size = std::min(left_idx_col->length(), right_idx_col->length());
      probe_func_ptr = &probe_hash_map_no_fill<TwoArrayIndexHashMMap>;
      goto init_ptrs;
    }
    case config::FULL_OUTER: {
      init_vec_size = left_idx_col->length() + right_idx_col->length();
      probe_func_ptr = &probe_hash_map_outer<TwoArrayIndexHashMMap>;

      init_ptrs:
      if (left_idx_col->length() <= right_idx_col->length()) {
        build_idx = false; // i.e. build hash map from left table, probe from right
      } else {
        build_idx = true; // i.e. build hash map from right table, probe from left
      }
      break;
    }
  }

  // reserve space for index vectors
  row_indices[0]->reserve(init_vec_size);
  row_indices[1]->reserve(init_vec_size);

  // populate left hash table
  TwoArrayIndexHash hash(*arrays[build_idx], *arrays[!build_idx]);
  TwoArrayIndexEqualTo equal_to(*arrays[build_idx], *arrays[!build_idx]);

  TwoArrayIndexHashMMap hash_map((*arrays[build_idx])->length(), hash, equal_to);

  // build hashmap from corresponding table
  for (int64_t i = 0; i < (*arrays[build_idx])->length(); i++) {
    hash_map.emplace(i, i);
  }

  // probe
  probe_func_ptr(hash_map, (*arrays[build_idx])->length(), (*arrays[!build_idx])->length(),
                 *row_indices[build_idx], *row_indices[!build_idx]);

  return arrow::Status::OK();
}

arrow::Status HashJoin(const std::shared_ptr<arrow::Table> &ltab,
                       const std::shared_ptr<arrow::Table> &rtab,
                       const config::JoinConfig &config,
                       std::shared_ptr<arrow::Table> *joined_table,
                       arrow::MemoryPool *memory_pool) {
  // let's first combine chunks in both tables
  std::shared_ptr<arrow::Table> c_ltab(ltab);
  if (ltab->column(0)->num_chunks() > 1) {
    const auto &l_res = ltab->CombineChunks(memory_pool);
    if (!l_res.ok()) return l_res.status();
    c_ltab = l_res.ValueOrDie();
  }

  std::shared_ptr<arrow::Table> c_rtab(rtab);
  if (rtab->column(0)->num_chunks() > 1) {
    const auto &r_res = rtab->CombineChunks(memory_pool);
    if (!r_res.ok()) return r_res.status();
    c_rtab = r_res.ValueOrDie();
  }

  if (config.GetLeftColumnIdx().size() == config.GetRightColumnIdx().size() && config.GetLeftColumnIdx().size() == 1) {
    int left_idx = config.GetLeftColumnIdx()[0];
    int right_idx = config.GetRightColumnIdx()[0];

    std::vector<int64_t> left_indices, right_indices;

    RETURN_ARROW_STATUS_IF_FAILED(
        ArrayIndexHashJoin(cylon::util::GetChunkOrEmptyArray(c_ltab->column(left_idx), 0),
                           cylon::util::GetChunkOrEmptyArray(c_rtab->column(right_idx), 0),
                           config.GetType(), left_indices, right_indices))

    return cylon::join::util::build_final_table(left_indices, right_indices, c_ltab, c_rtab,
                                                config.GetLeftTableSuffix(), config.GetRightTableSuffix(),
                                                joined_table, memory_pool);
  } else {
    return MultiIndexHashJoin(ltab, rtab, config, joined_table, memory_pool);
  }
}

}
}