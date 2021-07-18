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

#ifndef CYLON_SRC_CYLON_INDEXING_BUILDER_H_
#define CYLON_SRC_CYLON_INDEXING_BUILDER_H_

#include <cylon/indexing/index.hpp>
#include <cylon/status.hpp>
#include <cylon/table.hpp>

namespace cylon {

class IndexUtil {

 public:

  static Status BuildArrowIndexFromArray(const IndexingType schema,
										 const std::shared_ptr<Table> &input,
										 const std::shared_ptr<arrow::Array> &index_array);

  static Status BuildArrowIndex(IndexingType schema,
								const std::shared_ptr<Table> &input,
								int index_column,
								bool drop,
								std::shared_ptr<Table> &output);

  static Status BuildArrowIndex(const IndexingType schema,
								const std::shared_ptr<Table> &input,
								const int index_column,
								std::shared_ptr<cylon::BaseArrowIndex> &index);

  static Status BuildArrowHashIndex(const std::shared_ptr<Table> &input,
									const int index_column,
									std::shared_ptr<cylon::BaseArrowIndex> &index);

  static Status BuildArrowLinearIndex(const std::shared_ptr<Table> &input,
									  const int index_column,
									  std::shared_ptr<cylon::BaseArrowIndex> &index);

  static Status BuildArrowRangeIndex(const std::shared_ptr<Table> &input,
									 std::shared_ptr<cylon::BaseArrowIndex> &index);

  template<class TYPE>
  static Status BuildArrowNumericHashIndexFromArrowArray(const std::shared_ptr<arrow::Array> &index_values,
														 arrow::MemoryPool *pool,
														 std::shared_ptr<cylon::BaseArrowIndex> &index) {
	index = std::make_shared<ArrowNumericHashIndex<TYPE>>(-1, index_values->length(), pool, index_values);
	index->SetIndexArray(index_values);
	return Status::OK();
  }

  template<class TYPE>
  static Status BuildArrowBinaryHashIndexFromArrowArray(const std::shared_ptr<arrow::Array> &index_values,
														arrow::MemoryPool *pool,
														std::shared_ptr<cylon::BaseArrowIndex> &index) {
	index = std::make_shared<ArrowBinaryHashIndex<TYPE>>(-1, index_values->length(), pool, index_values);
	index->SetIndexArray(index_values);
	return Status::OK();
  }

  static Status BuildArrowLinearIndexFromArrowArray(const std::shared_ptr<arrow::Array> &index_values,
													arrow::MemoryPool *pool,
													std::shared_ptr<cylon::BaseArrowIndex> &index) {
	index = std::make_shared<ArrowLinearIndex>(0, index_values->length(), pool, index_values);
	return Status::OK();
  }

  static Status BuildArrowRangeIndexFromArray(int64_t size,
											  arrow::MemoryPool *pool,
											  std::shared_ptr<cylon::BaseArrowIndex> &index);

  static Status BuildArrowHashIndexFromArray(const std::shared_ptr<arrow::Array> &index_values,
											 arrow::MemoryPool *pool,
											 std::shared_ptr<cylon::BaseArrowIndex> &index);

};
}

#endif //CYLON_SRC_CYLON_INDEXING_BUILDER_H_
