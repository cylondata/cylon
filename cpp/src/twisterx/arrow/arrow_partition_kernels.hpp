#ifndef TWISTERX_ARROW_PARTITION_KERNELS_H
#define TWISTERX_ARROW_PARTITION_KERNELS_H

#include <memory>
#include <vector>
#include <arrow/api.h>
#include <glog/logging.h>

#include "../util/murmur3.hpp"

namespace twisterx {

class ArrowPartitionKernel {
public:
  explicit ArrowPartitionKernel(std::shared_ptr<arrow::DataType> type,
    arrow::MemoryPool* pool, std::shared_ptr<std::vector<int>> targets) : type_(type), pool_(pool), targets_(targets) {}

  /**
   * We partition the table and return the indexes as an array
   * @param ctx
   * @param values
   * @param out
   * @return
   */
  virtual int Partition(const std::shared_ptr <arrow::Array> &values,
                        std::shared_ptr<std::vector<int64_t>> partitions) = 0;
protected:
  std::shared_ptr<arrow::DataType> type_;
  arrow::MemoryPool* pool_;
  std::shared_ptr<std::vector<int>> targets_;
};

template <typename TYPE>
class NumericHashPartitionKernel : public ArrowPartitionKernel {
public:
  explicit NumericHashPartitionKernel(std::shared_ptr<arrow::DataType> type, arrow::MemoryPool* pool,
      std::shared_ptr<std::vector<int>> targets) : ArrowPartitionKernel(type, pool, targets) {}

  int Partition(const std::shared_ptr <arrow::Array> &values,
                std::shared_ptr<std::vector<int64_t>> partitions) override {
    auto reader = std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
    auto type = std::static_pointer_cast<arrow::FixedWidthType>(values->type());
    std::shared_ptr<arrow::Buffer> indices_buf;
    int bitWidth = type->bit_width();
    for (int64_t i = 0; i < reader->length(); i++) {
      auto lValue = reader->Value(i);
      void *val = (void *)&(lValue);
      uint32_t hash = 0;
      uint32_t seed = 0;
      // do the hash as we know the bit width
      twisterx::util::MurmurHash3_x86_32(val, bitWidth, seed, &hash);
      // this is the hash
      partitions->push_back(targets_->at(hash % targets_->size()));
    }
    // now build the
    return 0;
  }
};

using UInt8ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt8Type>;
using UInt16ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt16Type>;
using UInt32ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt32Type>;
using UInt64ArrayHashPartitioner = NumericHashPartitionKernel<arrow::UInt64Type>;
using Int8ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int8Type>;
using Int16ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int16Type>;
using Int32ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int32Type>;
using Int64ArrayHashPartitioner = NumericHashPartitionKernel<arrow::Int64Type>;
using HalfFloatArrayHashPartitioner = NumericHashPartitionKernel<arrow::HalfFloatType>;
using FloatArrayHashPartitioner = NumericHashPartitionKernel<arrow::FloatType>;
using DoubleArrayHashPartitioner = NumericHashPartitionKernel<arrow::DoubleType>;

arrow::Status HashPartitionArray(std::shared_ptr<arrow::DataType>& type, arrow::MemoryPool *pool,
                                 std::shared_ptr<arrow::Array> values,
                                 std::shared_ptr<std::vector<int>> targets,
                                 std::shared_ptr<std::vector<int64_t>> outPartitions);

}

#endif //TWISTERX_ARROW_PARTITION_KERNELS_H
