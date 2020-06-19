#include "arrow_memory_pool_utils.h"

arrow::Status twisterx::ArrowStatus(twisterx::Status status) {
  return arrow::Status(static_cast<arrow::StatusCode>(status.get_code()), status.get_msg());
}
arrow::MemoryPool *twisterx::ToArrowPool(twisterx::TwisterXContext *ctx) {
  if (ctx->GetMemoryPool() == nullptr) {
    return arrow::default_memory_pool();
  } else {
    return new ProxyMemoryPool(ctx->GetMemoryPool());
  }
}
