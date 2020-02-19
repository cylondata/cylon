#include "arrow_kernels.h"

namespace twisterx {
  int CreateNumericMerge(std::shared_ptr<arrow::DataType>& type,
                         arrow::MemoryPool* pool, std::shared_ptr<std::vector<int>> targets) {
    UInt8ArrayMerger(type, nullptr, targets);
    return 0;
  }
}