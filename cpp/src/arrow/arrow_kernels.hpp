#ifndef TWISTERX_ARROW_KERNELS_H
#define TWISTERX_ARROW_KERNELS_H

#include <arrow/api.h>
#include <arrow/compute/kernel.h>
#include <glog/logging.h>

namespace twisterx {
  class ArrowPartitionKernel {
    /**
     * We partition the table and return the indexes as an array
     * @param ctx
     * @param values
     * @param out
     * @return
     */
    virtual int Partition(arrow::compute::FunctionContext* ctx, const arrow::Table& values,
                          arrow::Int32Array* out) = 0;
  };

  class ArrowTableMergeKernel {
  public:
    explicit ArrowTableMergeKernel() {}

    /**
     * Merge the values in the column and return an array
     * @param ctx
     * @param values
     * @param targets
     * @param out_length
     * @param out
     * @return
     */
    virtual int Merge(arrow::compute::FunctionContext* ctx, const arrow::Table& values,
                          const arrow::Int32Array& partitions,
                          std::unordered_map<int, std::shared_ptr<arrow::Table>>& out) = 0;
  };

  class ArrowArrayMergeKernel {
  public:
    explicit ArrowArrayMergeKernel(std::shared_ptr<arrow::DataType> type,
        arrow::MemoryPool* pool, std::shared_ptr<std::vector<int>> targets) : type_(type), pool_(pool),
        targets_(targets) {}

    /**
     * Merge the values in the column and return an array
     * @param ctx
     * @param values
     * @param targets
     * @param out_length
     * @param out
     * @return
     */
    virtual int Merge(std::shared_ptr<arrow::Array> &values,
                      std::shared_ptr <arrow::Int32Array>& partitions,
                      std::unordered_map<int, std::shared_ptr<arrow::Array>>& out) = 0;
  protected:
    std::shared_ptr<arrow::DataType> type_;
    arrow::MemoryPool* pool_;
    std::shared_ptr<std::vector<int>> targets_;
  };

  template <typename TYPE>
  class ArrowArrayNumericMergeKernel : public ArrowArrayMergeKernel {
  public:
    explicit ArrowArrayNumericMergeKernel(std::shared_ptr<arrow::DataType> type,
                                   arrow::MemoryPool* pool, std::shared_ptr<std::vector<int>> targets) :
                                   ArrowArrayMergeKernel(type, pool, targets) {}

    int Merge(std::shared_ptr<arrow::Array> &values,
        std::shared_ptr <arrow::Int32Array>& partitions,
                      std::unordered_map<int, std::shared_ptr<arrow::Array>>& out) override {
      auto reader =
          std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
      std::unordered_map<int, std::shared_ptr<arrow::NumericBuilder<TYPE>>> builders;

      for (auto it = targets_.get()->begin() ; it != targets_.get()->end(); ++it) {
        std::shared_ptr<arrow::NumericBuilder<TYPE>> b = std::make_shared<arrow::NumericBuilder<TYPE>>(type_, pool_);
        builders.insert(std::pair<int, std::shared_ptr<arrow::NumericBuilder<TYPE>>>(*it, b));
      }

      for (int64_t i = 0; i < partitions->length(); i++) {
        std::shared_ptr<arrow::NumericBuilder<TYPE>> b = builders[partitions->Value(i)];
        b->Append(reader->Value(i));
      }

      for (auto it = targets_.get()->begin() ; it != targets_.get()->end(); ++it) {
        std::shared_ptr<arrow::NumericBuilder<TYPE>> b = builders[*it];
        std::shared_ptr<arrow::Array> array;
        b->Finish(&array);
        out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(*it, array));
      }
      return 0;
    }
  };

  using UInt8ArrayMerger = ArrowArrayNumericMergeKernel<arrow::UInt8Type>;
  using UInt16ArrayMerger = ArrowArrayNumericMergeKernel<arrow::UInt16Type>;
  using UInt32ArrayMerger = ArrowArrayNumericMergeKernel<arrow::UInt32Type>;
  using UInt64ArrayMerger = ArrowArrayNumericMergeKernel<arrow::UInt64Type>;

  using Int8ArrayMerger = ArrowArrayNumericMergeKernel<arrow::Int8Type>;
  using Int16ArrayMerger = ArrowArrayNumericMergeKernel<arrow::Int16Type>;
  using Int32ArrayMerger = ArrowArrayNumericMergeKernel<arrow::Int32Type>;
  using Int64ArrayMerger = ArrowArrayNumericMergeKernel<arrow::Int64Type>;

  using HalfFloatArrayMerger = ArrowArrayNumericMergeKernel<arrow::HalfFloatType>;
  using FloatArrayMerger = ArrowArrayNumericMergeKernel<arrow::FloatType>;
  using DoubleArrayMerger = ArrowArrayNumericMergeKernel<arrow::DoubleType>;

  int CreateNumericMerge(std::shared_ptr<arrow::DataType>& type,
                         arrow::MemoryPool* pool, std::shared_ptr<std::vector<int>> targets,
                         std::unique_ptr<ArrowArrayMergeKernel>* out);
}

#endif //TWISTERX_ARROW_KERNELS_H
