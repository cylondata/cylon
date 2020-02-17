#ifndef TWISTERX_ARROW_KERNELS_H
#define TWISTERX_ARROW_KERNELS_H

#include <arrow/compute/kernel.h>

namespace twisterx {
  class ArrowMergeKernel {
  public:
    explicit ArrowMergeKernel() {}

    /**
     * We partition the table and return the indexes as an array
     * @param ctx
     * @param values
     * @param out
     * @return
     */
    virtual int Partition(arrow::compute::FunctionContext* ctx, const arrow::Table& values,
                arrow::Int32Array* out) = 0;


    /**
     * Merge the values in the colum and return an array
     * @param ctx
     * @param values
     * @param targets
     * @param out_length
     * @param out
     * @return
     */
    virtual int Merge(arrow::compute::FunctionContext* ctx, const arrow::Array& values,
                          const arrow::Int32Array& targets, int32_t columnIndex,
                          std::shared_ptr<arrow::Array>* out) = 0;
  protected:
    std::shared_ptr<arrow::DataType> type_;
  };
}


#endif //TWISTERX_ARROW_KERNELS_H
