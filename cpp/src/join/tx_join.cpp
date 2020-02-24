#include "tx_join.hpp"
#include "arrow/compute/api.h"
#include <glog/logging.h>
#include <chrono>
#include <map>

namespace twisterx::join {

template<typename JOIN_COLUMN_ARRAY, typename ARROW_KEY_TYPE, typename CPP_KEY_TYPE,
	typename = std::enable_if<std::is_base_of<arrow::NumericArray<ARROW_KEY_TYPE>, JOIN_COLUMN_ARRAY>::value>>
void advance(std::vector<CPP_KEY_TYPE> *subset,
			 std::shared_ptr<arrow::Int64Array> sorted_indices, // this is always Int64Array
			 int64_t *current_index, //always int32_t
			 std::shared_ptr<JOIN_COLUMN_ARRAY> data_column,
			 CPP_KEY_TYPE *key) {
  subset->clear();
  if (*current_index == sorted_indices->length()) {
	return;
  }
  int64_t data_index = sorted_indices->Value(*current_index);
  *key = data_column->Value(data_index);
  while (*current_index < sorted_indices->length() && data_column->Value(data_index) == *key) {
	subset->push_back(data_index);
	(*current_index)++;
	if (*current_index == sorted_indices->length()) {
	  break;
	}
	data_index = sorted_indices->Value(*current_index);
  }
}

void build_final_table() {

}

template<typename JOIN_COLUMN_ARRAY, typename ARROW_KEY_TYPE, typename CPP_KEY_TYPE>
void join(std::shared_ptr<arrow::Table> left_tab, std::shared_ptr<arrow::Table> right_tab,
		  int64_t left_join_column_idx,
		  int64_t right_join_column_idx,
		  std::shared_ptr<std::unordered_map<int64_t, arrow::ArrayBuilder>> column_builders,
		  std::shared_ptr<std::unordered_map<int64_t, arrow::ArrayBuilder>> right_builders,
		  JoinType join_type,
		  JoinAlgorithm join_algorithm) {
  //sort columns
  auto left_join_column = std::static_pointer_cast<JOIN_COLUMN_ARRAY>(left_tab->column(left_join_column_idx)->chunk(0));
  auto right_join_column =
	  std::static_pointer_cast<JOIN_COLUMN_ARRAY>(right_tab->column(right_join_column_idx)->chunk(0));

  auto t1 = std::chrono::high_resolution_clock::now();
  arrow::compute::FunctionContext ctx_left;
  std::shared_ptr<arrow::Array> left_index_sorted_column;
  auto status = arrow::compute::SortToIndices(&ctx_left, *left_join_column, &left_index_sorted_column);
  if (status != arrow::Status::OK()) {
	// error
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "left sorting time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  t1 = std::chrono::high_resolution_clock::now();
  arrow::compute::FunctionContext ctx;
  std::shared_ptr<arrow::Array> right_index_sorted_column;
  status = arrow::compute::SortToIndices(&ctx, *right_join_column, &right_index_sorted_column);
  if (status != arrow::Status::OK()) {
	// error
  }
  t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "right sorting time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  // creating joined schema
  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.insert(fields.end(), left_tab->schema()->fields().begin(), left_tab->schema()->fields().end());
  auto schema = arrow::schema(fields);

  CPP_KEY_TYPE left_key, right_key;
  std::vector<CPP_KEY_TYPE> left_subset, right_subset;
  int64_t left_current_index = 0;
  int64_t right_current_index = 0;

  std::map<int64_t, std::vector<int64_t >> join_relations; // using map intentionally to keep elements ordered

  t1 = std::chrono::high_resolution_clock::now();

  advance<JOIN_COLUMN_ARRAY, ARROW_KEY_TYPE, CPP_KEY_TYPE>(&left_subset,
														   std::static_pointer_cast<arrow::Int64Array>(
															   left_index_sorted_column),
														   &left_current_index,
														   left_join_column,
														   &left_key);

  advance<JOIN_COLUMN_ARRAY, ARROW_KEY_TYPE, CPP_KEY_TYPE>(&right_subset,
														   std::static_pointer_cast<arrow::Int64Array>(
															   right_index_sorted_column),
														   &right_current_index,
														   right_join_column,
														   &right_key);
  while (!left_subset.empty() && !right_subset.empty()) {
	if (left_key == right_key) { // use a key comparator
	  for (int64_t left_idx: left_subset) {
		std::vector<int64_t> right_mappings;
		for (int64_t right_idx: right_subset) {
		  right_mappings.push_back(right_idx);
		}
		join_relations.insert(std::pair(left_idx, right_mappings));

		//advance
		advance<JOIN_COLUMN_ARRAY, ARROW_KEY_TYPE, CPP_KEY_TYPE>(&left_subset,
																 std::static_pointer_cast<arrow::Int64Array>(
																	 left_index_sorted_column),
																 &left_current_index,
																 left_join_column,
																 &left_key);

		advance<JOIN_COLUMN_ARRAY, ARROW_KEY_TYPE, CPP_KEY_TYPE>(&right_subset,
																 std::static_pointer_cast<arrow::Int64Array>(
																	 right_index_sorted_column),
																 &right_current_index,
																 right_join_column,
																 &right_key);
	  }
	} else if (left_key < right_key) {
	  advance<JOIN_COLUMN_ARRAY, ARROW_KEY_TYPE, CPP_KEY_TYPE>(&left_subset,
															   std::static_pointer_cast<arrow::Int64Array>(
																   left_index_sorted_column),
															   &left_current_index,
															   left_join_column,
															   &left_key);
	} else {
	  advance<JOIN_COLUMN_ARRAY, ARROW_KEY_TYPE, CPP_KEY_TYPE>(&right_subset,
															   std::static_pointer_cast<arrow::Int64Array>(
																   right_index_sorted_column),
															   &right_current_index,
															   right_join_column,
															   &right_key);
	}
  }
  t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "join only time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "done and produced : " << join_relations.size();
}

template<typename JOIN_COLUMN_ARRAY, typename ARROW_KEY_TYPE, typename CPP_KEY_TYPE>
void join(std::vector<std::shared_ptr<arrow::Table>>& left_tabs,
		  std::vector<std::shared_ptr<arrow::Table>>& right_tabs,
		  int64_t left_join_column_idx,
		  int64_t right_join_column_idx,
		  std::shared_ptr<std::unordered_map<int64_t, arrow::ArrayBuilder>> column_builders,
		  std::shared_ptr<std::unordered_map<int64_t, arrow::ArrayBuilder>> right_builders,
		  JoinType join_type,
		  JoinAlgorithm join_algorithm,
		  arrow::MemoryPool *memory_pool) {
  std::shared_ptr<arrow::Table> left_tab = arrow::ConcatenateTables(left_tabs,
																	arrow::ConcatenateTablesOptions::Defaults(),
																	memory_pool).ValueOrDie();
  std::shared_ptr<arrow::Table> right_tab = arrow::ConcatenateTables(right_tabs,
																	 arrow::ConcatenateTablesOptions::Defaults(),
																	 memory_pool).ValueOrDie();
  std::shared_ptr<arrow::Table> left_tab_combined;
  std::shared_ptr<arrow::Table> right_tab_combined;
  arrow::Status left_combine_stat = left_tab->CombineChunks(memory_pool, &left_tab_combined);
  arrow::Status right_combine_stat = right_tab->CombineChunks(memory_pool, &right_tab_combined);

  if (left_combine_stat == arrow::Status::OK() && right_combine_stat == arrow::Status::OK()) {
	twisterx::join::join<JOIN_COLUMN_ARRAY, ARROW_KEY_TYPE, CPP_KEY_TYPE>(left_tab_combined,
																		  right_tab_combined,
																		  left_join_column_idx,
																		  right_join_column_idx,
																		  column_builders,
																		  right_builders,
																		  join_type,
																		  join_algorithm);
  } else {
	LOG(ERROR) << "Error in combining table chunks. Aborting join operation...";
  }
}
}