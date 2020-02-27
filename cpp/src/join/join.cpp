#include "join.hpp"
#include "arrow/compute/api.h"
#include <glog/logging.h>
#include <chrono>
#include <map>
#include "join_utils.hpp"

namespace twisterx::join {

template<typename ARROW_KEY_TYPE, typename CPP_KEY_TYPE>
void advance(std::vector<CPP_KEY_TYPE> *subset,
			 const std::shared_ptr<arrow::Int64Array> &sorted_indices, // this is always Int64Array
			 int64_t *current_index, //always int64_t
			 std::shared_ptr<arrow::Array> data_column,
			 CPP_KEY_TYPE *key) {
  subset->clear();
  if (*current_index == sorted_indices->length()) {
	return;
  }
  auto data_column_casted = std::static_pointer_cast<arrow::NumericArray<ARROW_KEY_TYPE>>(data_column);
  int64_t data_index = sorted_indices->Value(*current_index);
  *key = data_column_casted->Value(data_index);
  while (*current_index < sorted_indices->length() && data_column_casted->Value(data_index) == *key) {
	subset->push_back(data_index);
	(*current_index)++;
	if (*current_index == sorted_indices->length()) {
	  break;
	}
	data_index = sorted_indices->Value(*current_index);
  }
}

template<typename ARROW_KEY_TYPE, typename CPP_KEY_TYPE>
std::shared_ptr<arrow::Table> do_sorted_inner_join(const std::shared_ptr<arrow::Table> &left_tab,
												   const std::shared_ptr<arrow::Table> &right_tab,
												   int64_t left_join_column_idx,
												   int64_t right_join_column_idx,
												   arrow::MemoryPool *memory_pool) {
  //sort columns
  auto left_join_column = left_tab->column(left_join_column_idx)->chunk(0);
  auto right_join_column = right_tab->column(right_join_column_idx)->chunk(0);

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
  // Done sorting columns

  CPP_KEY_TYPE left_key, right_key;
  std::vector<CPP_KEY_TYPE> left_subset, right_subset;
  int64_t left_current_index = 0;
  int64_t right_current_index = 0;

  std::map<int64_t, std::vector<int64_t >> join_relations; // using map intentionally to keep elements ordered

  t1 = std::chrono::high_resolution_clock::now();

  advance<ARROW_KEY_TYPE, CPP_KEY_TYPE>(&left_subset,
										std::static_pointer_cast<arrow::Int64Array>(
											left_index_sorted_column),
										&left_current_index,
										left_join_column,
										&left_key);

  advance<ARROW_KEY_TYPE, CPP_KEY_TYPE>(&right_subset,
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
		join_relations.insert(std::pair<int64_t, std::vector<int64_t >>(left_idx, right_mappings));

		//advance
		advance<ARROW_KEY_TYPE, CPP_KEY_TYPE>(&left_subset,
											  std::static_pointer_cast<arrow::Int64Array>(
												  left_index_sorted_column),
											  &left_current_index,
											  left_join_column,
											  &left_key);

		advance<ARROW_KEY_TYPE, CPP_KEY_TYPE>(&right_subset,
											  std::static_pointer_cast<arrow::Int64Array>(
												  right_index_sorted_column),
											  &right_current_index,
											  right_join_column,
											  &right_key);
	  }
	} else if (left_key < right_key) {
	  advance<ARROW_KEY_TYPE, CPP_KEY_TYPE>(&left_subset,
											std::static_pointer_cast<arrow::Int64Array>(
												left_index_sorted_column),
											&left_current_index,
											left_join_column,
											&left_key);
	} else {
	  advance<ARROW_KEY_TYPE, CPP_KEY_TYPE>(&right_subset,
											std::static_pointer_cast<arrow::Int64Array>(
												right_index_sorted_column),
											&right_current_index,
											right_join_column,
											&right_key);
	}
  }

  t2 = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "index join time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  LOG(INFO) << "building final table...'";

  t1 = std::chrono::high_resolution_clock::now();

  // build final table
  std::shared_ptr<arrow::Table> final_table = twisterx::join::util::build_final_table(
	  std::make_shared<std::map<int64_t, std::vector<int64_t >>>(join_relations),
	  left_tab,
	  right_tab,
	  memory_pool
  );

  t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "built final table in : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "done and produced : " << join_relations.size();

  return final_table;
}

template<typename ARROW_KEY_TYPE, typename CPP_KEY_TYPE>
std::shared_ptr<arrow::Table> do_join(const std::shared_ptr<arrow::Table> &left_tab,
									  const std::shared_ptr<arrow::Table> &right_tab,
									  int64_t left_join_column_idx,
									  int64_t right_join_column_idx,
									  JoinType join_type,
									  JoinAlgorithm join_algorithm,
									  arrow::MemoryPool *memory_pool) {
  if (join_type == JoinType::INNER) {
	switch (join_algorithm) {
	  case SORT:
		return do_sorted_inner_join<ARROW_KEY_TYPE, CPP_KEY_TYPE>(left_tab,
																  right_tab,
																  left_join_column_idx,
																  right_join_column_idx, memory_pool);
	  case HASH:break;
	}
  }
}

std::shared_ptr<arrow::Table> join(const std::vector<std::shared_ptr<arrow::Table>> &left_tabs,
								   const std::vector<std::shared_ptr<arrow::Table>> &right_tabs,
								   int64_t left_join_column_idx,
								   int64_t right_join_column_idx,
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
	return twisterx::join::join(left_tab_combined,
								right_tab_combined,
								left_join_column_idx,
								right_join_column_idx,
								join_type,
								join_algorithm, memory_pool);
  } else {
	LOG(ERROR) << "Error in combining table chunks. Aborting join operation...";
  }
}

std::shared_ptr<arrow::Table> join(const std::shared_ptr<arrow::Table> &left_tab,
								   const std::shared_ptr<arrow::Table> &right_tab,
								   int64_t left_join_column_idx,
								   int64_t right_join_column_idx,
								   JoinType join_type,
								   JoinAlgorithm join_algorithm,
								   arrow::MemoryPool *memory_pool) {
  auto left_type = left_tab->column(left_join_column_idx)->type()->id();
  auto right_type = right_tab->column(right_join_column_idx)->type()->id();

  if (left_type != right_type) {
	//fail
  }

  switch (left_type) {
	case arrow::Type::NA:break;
	case arrow::Type::BOOL:break;
	case arrow::Type::UINT8:
	  return do_join<arrow::UInt8Type, int8_t>(left_tab,
											   right_tab,
											   left_join_column_idx,
											   right_join_column_idx, join_type, join_algorithm, memory_pool);
	case arrow::Type::INT8:
	  return do_join<arrow::Int8Type, int8_t>(left_tab,
											  right_tab,
											  left_join_column_idx,
											  right_join_column_idx, join_type, join_algorithm, memory_pool);
	case arrow::Type::UINT16:
	  return do_join<arrow::UInt16Type, uint16_t>(left_tab,
												  right_tab,
												  left_join_column_idx,
												  right_join_column_idx, join_type, join_algorithm, memory_pool);
	case arrow::Type::INT16:
	  return do_join<arrow::Int16Type, int16_t>(left_tab,
												right_tab,
												left_join_column_idx,
												right_join_column_idx, join_type, join_algorithm, memory_pool);
	case arrow::Type::UINT32:
	  return do_join<arrow::UInt32Type, uint32_t>(left_tab,
												  right_tab,
												  left_join_column_idx,
												  right_join_column_idx, join_type, join_algorithm, memory_pool);
	case arrow::Type::INT32:
	  return do_join<arrow::Int32Type, int32_t>(left_tab,
												right_tab,
												left_join_column_idx,
												right_join_column_idx, join_type, join_algorithm, memory_pool);
	case arrow::Type::UINT64:
	  return do_join<arrow::UInt64Type, uint64_t>(left_tab,
												  right_tab,
												  left_join_column_idx,
												  right_join_column_idx, join_type, join_algorithm, memory_pool);
	case arrow::Type::INT64:
	  return do_join<arrow::Int64Type, int64_t>(left_tab,
												right_tab,
												left_join_column_idx,
												right_join_column_idx, join_type, join_algorithm, memory_pool);;
	case arrow::Type::HALF_FLOAT:
	  return do_join<arrow::HalfFloatType, uint16_t>(left_tab,
													 right_tab,
													 left_join_column_idx,
													 right_join_column_idx, join_type, join_algorithm, memory_pool);
	case arrow::Type::FLOAT:
	  return do_join<arrow::FloatType, float_t>(left_tab,
												right_tab,
												left_join_column_idx,
												right_join_column_idx, join_type, join_algorithm, memory_pool);
	case arrow::Type::DOUBLE:
	  return do_join<arrow::DoubleType, double_t>(left_tab,
												  right_tab,
												  left_join_column_idx,
												  right_join_column_idx, join_type, join_algorithm, memory_pool);
	case arrow::Type::STRING:break;
	case arrow::Type::BINARY:break;
	case arrow::Type::FIXED_SIZE_BINARY:break;
	case arrow::Type::DATE32:break;
	case arrow::Type::DATE64:break;
	case arrow::Type::TIMESTAMP:break;
	case arrow::Type::TIME32:break;
	case arrow::Type::TIME64:break;
	case arrow::Type::INTERVAL:break;
	case arrow::Type::DECIMAL:break;
	case arrow::Type::LIST:break;
	case arrow::Type::STRUCT:break;
	case arrow::Type::UNION:break;
	case arrow::Type::DICTIONARY:break;
	case arrow::Type::MAP:break;
	case arrow::Type::EXTENSION:break;
	case arrow::Type::FIXED_SIZE_LIST:break;
	case arrow::Type::DURATION:break;
	case arrow::Type::LARGE_STRING:break;
	case arrow::Type::LARGE_BINARY:break;
	case arrow::Type::LARGE_LIST:break;
  }
}
}