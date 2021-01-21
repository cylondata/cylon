#include "index.hpp"
#include "table.hpp"

namespace cylon {

std::unique_ptr<IndexKernel> CreateHashIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column) {
  switch (input_table->column(index_column)->chunk(0)->type()->id()) {

    case arrow::Type::NA:return nullptr;
    case arrow::Type::BOOL:return std::make_unique<BoolHashIndexKernel>();
    case arrow::Type::UINT8:return std::make_unique<UInt8HashIndexKernel>();
    case arrow::Type::INT8:return std::make_unique<Int8HashIndexKernel>();
    case arrow::Type::UINT16:return std::make_unique<UInt16HashIndexKernel>();
    case arrow::Type::INT16:return std::make_unique<Int16HashIndexKernel>();
    case arrow::Type::UINT32:return std::make_unique<UInt32HashIndexKernel>();
    case arrow::Type::INT32:return std::make_unique<Int32HashIndexKernel>();
    case arrow::Type::UINT64:return std::make_unique<UInt64HashIndexKernel>();
    case arrow::Type::INT64: return std::make_unique<Int64HashIndexKernel>();
    case arrow::Type::HALF_FLOAT:return std::make_unique<HalfFloatHashIndexKernel>();
    case arrow::Type::FLOAT:return std::make_unique<FloatHashIndexKernel>();
    case arrow::Type::DOUBLE:return std::make_unique<DoubleHashIndexKernel>();
    case arrow::Type::STRING:return std::make_unique<StringHashIndexKernel>();
    case arrow::Type::BINARY:return nullptr;//std::make_unique<BinaryHashIndexKernel>();
    default: return std::make_unique<GenericRangeIndexKernel>();
  }

}

std::unique_ptr<IndexKernel> CreateLinearIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column) {
  switch (input_table->column(index_column)->chunk(0)->type()->id()) {

    case arrow::Type::NA:return nullptr;
    case arrow::Type::BOOL:return std::make_unique<BoolLinearIndexKernel>();
    case arrow::Type::UINT8:return std::make_unique<UInt8LinearIndexKernel>();
    case arrow::Type::INT8:return std::make_unique<Int8LinearIndexKernel>();
    case arrow::Type::UINT16:return std::make_unique<UInt16LinearIndexKernel>();
    case arrow::Type::INT16:return std::make_unique<Int16LinearIndexKernel>();
    case arrow::Type::UINT32:return std::make_unique<UInt32LinearIndexKernel>();
    case arrow::Type::INT32:return std::make_unique<Int32LinearIndexKernel>();
    case arrow::Type::UINT64:return std::make_unique<UInt64LinearIndexKernel>();
    case arrow::Type::INT64: return std::make_unique<Int64LinearIndexKernel>();
    case arrow::Type::HALF_FLOAT:return std::make_unique<HalfFloatLinearIndexKernel>();
    case arrow::Type::FLOAT:return std::make_unique<FloatLinearIndexKernel>();
    case arrow::Type::DOUBLE:return std::make_unique<DoubleLinearIndexKernel>();
    case arrow::Type::STRING:return std::make_unique<StringLinearIndexKernel>();
    case arrow::Type::BINARY:return nullptr;//std::make_unique<BinaryLinearIndexKernel>();
    default: return std::make_unique<GenericRangeIndexKernel>();
  }

}


std::unique_ptr<IndexKernel> CreateIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column) {
  // TODO: fix the criterion to check the kernel creation method
  if (index_column == -1) {
    return std::make_unique<GenericRangeIndexKernel>();
  } else {
    return CreateHashIndexKernel(input_table, index_column);
  }
}
cylon::RangeIndex::RangeIndex(int start, int size, int step, arrow::MemoryPool *pool) : BaseIndex(0, size, pool),
                                                                                        start_(start),
                                                                                        end_(size),
                                                                                        step_(step) {

}
Status RangeIndex::LocationByValue(void *search_param,
                                   std::shared_ptr<arrow::Table> &input,
                                   std::vector<int64_t> &filter_locations,
                                   std::shared_ptr<arrow::Table> &output) {
  LOG(ERROR) << "Not Implemented!";
  return Status::OK();
}

int RangeIndex::GetColId() const {
  return BaseIndex::GetColId();
}
int RangeIndex::GetSize() const {
  return BaseIndex::GetSize();
}
arrow::MemoryPool *RangeIndex::GetPool() const {
  return BaseIndex::GetPool();
}
int RangeIndex::GetStart() const {
  return start_;
}
int RangeIndex::GetAnEnd() const {
  return end_;
}
int RangeIndex::GetStep() const {
  return step_;
}
Status RangeIndex::LocationByValue(void *search_param, std::vector<int64_t> &find_index) {
  int64_t val = *static_cast<int64_t *>(search_param);
  find_index.push_back(val);
  return Status::OK();
}
Status RangeIndex::LocationByValue(void *search_param, int64_t &find_index) {
  find_index = *static_cast<int64_t *>(search_param);
  return Status::OK();
}
std::shared_ptr<arrow::Array> RangeIndex::GetIndexAsArray() {

  arrow::Status arrow_status;
  auto pool = GetPool();

  arrow::Int64Builder builder(pool);

  std::shared_ptr<arrow::Int64Array> index_array;

  std::vector<int64_t> vec(GetSize(), 1);

  for (int64_t ix = 0; ix < GetSize(); ix += GetStep()) {
    vec[ix] = ix;
  }

  builder.AppendValues(vec);
  arrow_status = builder.Finish(&index_array);

  if (!arrow_status.ok()) {
    LOG(ERROR) << "Error occurred in retrieving index";
    return nullptr;
  }

  return index_array;
}
void RangeIndex::SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) {
  index_arr_ = index_arr;
}
std::shared_ptr<arrow::Array> RangeIndex::GetIndexArray() {
  return index_arr_;
}

RangeIndexKernel::RangeIndexKernel() {}

std::shared_ptr<BaseIndex> RangeIndexKernel::BuildIndex(arrow::MemoryPool *pool,
                                                        std::shared_ptr<arrow::Table> &input_table,
                                                        const int index_column) {
  std::shared_ptr<RangeIndex> range_index;

  range_index = std::make_shared<RangeIndex>(0, input_table->num_rows(), 1, pool);

  return range_index;
}

int BaseIndex::GetColId() const {
  return col_id_;
}
int BaseIndex::GetSize() const {
  return size_;
}
arrow::MemoryPool *BaseIndex::GetPool() const {
  return pool_;
}



}





