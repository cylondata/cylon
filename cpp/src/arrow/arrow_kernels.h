#ifndef TWISTERX_ARROW_KERNELS_H
#define TWISTERX_ARROW_KERNELS_H

#include <arrow/compute/kernel.h>

namespace twisterx {

  class ArrowMergeKernel {
  public:
    explicit ArrowMergeKernel(const std::shared_ptr<arrow::DataType>& type) : type_(type) {}

    /// \brief BinaryKernel interface
    ///
    /// delegates to subclasses via Filter()
    virtual int Call(arrow::compute::FunctionContext* ctx, const arrow::compute::Datum& values, const arrow::compute::Datum& filter,
                arrow::compute::Datum* out) = 0;

    /// \brief output type of this kernel (identical to type of values filtered)
    std::shared_ptr<arrow::DataType> out_type() const { return type_; }

    /// \brief factory for FilterKernels
    ///
    /// \param[in] value_type constructed FilterKernel will support filtering
    ///            values of this type
    /// \param[out] out created kernel
    static int Make(const std::shared_ptr<arrow::DataType>& value_type,
                       std::unique_ptr<arrow::compute::BinaryKernel>* out) { return 0; }

    /// \brief single-array implementation
    virtual int Merge(arrow::compute::FunctionContext* ctx, const arrow::Array& values,
                          const arrow::BooleanArray& filter, int64_t out_length,
                          std::shared_ptr<arrow::Array>* out) = 0;

  protected:
    std::shared_ptr<arrow::DataType> type_;
  };

}


#endif //TWISTERX_ARROW_KERNELS_H
