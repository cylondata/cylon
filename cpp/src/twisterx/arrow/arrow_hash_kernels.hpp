//
// Created by nira on 5/21/20.
//

#ifndef TWISTERX_CPP_SRC_TWISTERX_ARROW_ARROW_HASH_KERNELS_HPP_
#define TWISTERX_CPP_SRC_TWISTERX_ARROW_ARROW_HASH_KERNELS_HPP_

#include <arrow/api.h>
#include <arrow/compute/kernel.h>
#include <glog/logging.h>
#include "../status.hpp"
#include "../join/join_config.h"
#include "iostream"
#include <unordered_set>

namespace twisterx {

// default implementation for Numeric arrow types
template<class ARROW_TYPE>
class ArrowArrayIdxHashJoinKernel {
 public:
  using CTYPE = typename ARROW_TYPE::c_type;
  using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;

  virtual int BuildHashMap(const std::shared_ptr<arrow::Array> &left_idx_col,
						   const std::shared_ptr<arrow::Array> &right_idx_col,
						   const twisterx::join::config::JoinType join_type,
						   std::shared_ptr<std::vector<int64_t>> &left_output,
						   std::shared_ptr<std::vector<int64_t>> &right_output) {

	std::unique_ptr<MMAP_TYPE> out_umm = std::make_unique<MMAP_TYPE>();

	switch (join_type) {
	  case twisterx::join::config::JoinType::RIGHT: {
		BuildPhase(left_idx_col, out_umm);
		ProbePhase(out_umm, right_idx_col, left_output, right_output);
		break;
	  }
	  case twisterx::join::config::JoinType::LEFT: {
		BuildPhase(right_idx_col, out_umm);
		ProbePhase(out_umm, left_idx_col, right_output, left_output);
		break;
	  }
	  case twisterx::join::config::JoinType::INNER: {
		if (left_idx_col->length() < right_idx_col->length()) {
		  BuildPhase(left_idx_col, out_umm);
		  ProbePhaseNoFill(out_umm, right_idx_col, left_output, right_output);
		} else {
		  BuildPhase(right_idx_col, out_umm);
		  ProbePhaseNoFill(out_umm, left_idx_col, right_output, left_output);
		}
		break;
	  }
	  case twisterx::join::config::JoinType::FULL_OUTER: {
		// todo: use an index vector rather than a key set!
		std::unique_ptr<std::unordered_set<CTYPE>> key_set = std::make_unique<std::unordered_set<CTYPE>>();

		if (left_idx_col->length() < right_idx_col->length()) {
		  BuildPhase(left_idx_col, out_umm, key_set);
		  ProbePhaseOuter(out_umm, right_idx_col, key_set, left_output, right_output);
		} else {
		  BuildPhase(right_idx_col, out_umm, key_set);
		  ProbePhaseOuter(out_umm, left_idx_col, key_set, right_output, left_output);
		}
		break;
	  }
	  default: LOG(ERROR) << "not implemented!";
	}
	return 0;
  }

 private:
  void BuildPhase(const std::shared_ptr<arrow::Array> &smaller_idx_col,
				  std::unique_ptr<MMAP_TYPE> &smaller_idx_map) {
	auto reader0 = std::static_pointer_cast<arrow::NumericArray<ARROW_TYPE>>(smaller_idx_col);

	for (int64_t i = 0; i < reader0->length(); i++) {
	  auto lValue = reader0->Value(i);
	  CTYPE val = (CTYPE)lValue;
	  smaller_idx_map->insert(std::make_pair(val, i));
	}
  }

  void BuildPhase(const std::shared_ptr<arrow::Array> &smaller_idx_col,
				  std::unique_ptr<MMAP_TYPE> &smaller_idx_map,
				  std::unique_ptr<std::unordered_set<CTYPE>> &smaller_key_set) {
	auto reader0 = std::static_pointer_cast<arrow::NumericArray<ARROW_TYPE>>(smaller_idx_col);

	for (int64_t i = 0; i < reader0->length(); i++) {
	  auto lValue = reader0->Value(i);
	  CTYPE val = (CTYPE)lValue;
	  smaller_idx_map->insert(std::make_pair(val, i));
	  smaller_key_set->emplace(val);
	}
  }

  void ProbePhase(const std::unique_ptr<MMAP_TYPE> &smaller_idx_map,
				  const std::shared_ptr<arrow::Array> &larger_idx_col,
				  std::shared_ptr<std::vector<int64_t>> &smaller_output,
				  std::shared_ptr<std::vector<int64_t>> &larger_output) {
	auto reader1 = std::static_pointer_cast<arrow::NumericArray<ARROW_TYPE>>(larger_idx_col);
	for (int64_t i = 0; i < reader1->length(); ++i) {
	  CTYPE val = (CTYPE)reader1->Value(i);
	  auto range = smaller_idx_map->equal_range(val);
	  if (range.first == range.second) {
		smaller_output->push_back(-1);
		larger_output->push_back(i);
	  } else {
		for (auto it = range.first; it != range.second; it++) {
		  smaller_output->push_back(it->second);
		  larger_output->push_back(i);
		}
	  }
	}
  }

  void ProbePhaseNoFill(const std::unique_ptr<MMAP_TYPE> &smaller_idx_map,
						const std::shared_ptr<arrow::Array> &larger_idx_col,
						std::shared_ptr<std::vector<int64_t>> &smaller_output,
						std::shared_ptr<std::vector<int64_t>> &larger_output) {
	auto reader1 = std::static_pointer_cast<arrow::NumericArray<ARROW_TYPE>>(larger_idx_col);
	for (int64_t i = 0; i < reader1->length(); ++i) {
	  CTYPE val = (CTYPE)reader1->Value(i);
	  auto range = smaller_idx_map->equal_range(val);
	  for (auto it = range.first; it != range.second; it++) {
		smaller_output->push_back(it->second);
		larger_output->push_back(i);
	  }
	}
  }

  void ProbePhaseOuter(const std::unique_ptr<MMAP_TYPE> &smaller_idx_map,
					   const std::shared_ptr<arrow::Array> &larger_idx_col,
					   std::unique_ptr<std::unordered_set<CTYPE>> &smaller_key_set,
					   std::shared_ptr<std::vector<int64_t>> &smaller_output,
					   std::shared_ptr<std::vector<int64_t>> &larger_output) {
	auto reader1 = std::static_pointer_cast<arrow::NumericArray<ARROW_TYPE>>(larger_idx_col);
	for (int64_t i = 0; i < reader1->length(); ++i) {
	  CTYPE val = (CTYPE)reader1->Value(i);
	  auto range = smaller_idx_map->equal_range(val);
	  if (range.first == range.second) {
		smaller_output->push_back(-1);
		larger_output->push_back(i);
	  } else {
		for (auto it = range.first; it != range.second; it++) {
		  smaller_key_set->erase(it->first); // todo: this erase would be inefficient
		  smaller_output->push_back(it->second);
		  larger_output->push_back(i);
		}
	  }
	}

	// todo: use an index vector rather than a key set! this second probe is inefficient!
	for (auto it = smaller_key_set->begin(); it != smaller_key_set->end(); it++) {
	  auto range = smaller_idx_map->equal_range(*it);
	  for (auto it2 = range.first; it2 != range.second; it2++) {
		smaller_output->push_back(it2->second);
		larger_output->push_back(-1);
	  }
	}
  }
};
}
#endif //TWISTERX_CPP_SRC_TWISTERX_ARROW_ARROW_HASH_KERNELS_HPP_
