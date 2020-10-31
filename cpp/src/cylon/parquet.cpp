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
#include "parquet.hpp"

#include <arrow/table.h>
#include <memory>
#include <future>
#include "io/arrow_io.hpp"
#include "ctx/arrow_memory_pool_utils.hpp"

namespace cylon {
Status FromParquet(std::shared_ptr<cylon::CylonContext> &ctx, const std::string &path,
                   std::shared_ptr<Table> &tableOut) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = cylon::io::ReadParquet(ctx, path);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> table = *result;
    LOG(INFO) << "Chunks " << table->column(0)->chunks().size();
    if (table->column(0)->chunks().size() > 1) {
      auto status = table->CombineChunks(ToArrowPool(ctx), &table);
      if (!status.ok()) {
        return Status(Code::IOError, status.message());
      }
    }
    tableOut = std::make_shared<Table>(table, ctx);
    return Status(Code::OK, result.status().message());
  }
  return Status(Code::IOError, result.status().message());
}

void ReadParquetThread(const std::shared_ptr<CylonContext> &ctx, const std::string &path,
                       std::shared_ptr<cylon::Table> *table,
                       const std::shared_ptr<std::promise<Status>> &status_promise) {
  std::shared_ptr<CylonContext> ctx_ = ctx; // make a copy of the shared ptr
  status_promise->set_value(FromParquet(ctx_,
                                        path,
                                        *table));
}

Status FromParquet(std::shared_ptr<cylon::CylonContext> &ctx, const std::vector<std::string> &paths,
                   const std::vector<std::shared_ptr<Table> *> &tableOuts,
                   io::config::ParquetOptions options) {
  if (options.IsConcurrentFileReads()) {
    std::vector<std::pair<std::future<Status>, std::thread>> futures;
    futures.reserve(paths.size());
    for (uint64_t kI = 0; kI < paths.size(); ++kI) {
      auto read_promise = std::make_shared<std::promise<Status>>();
      futures.emplace_back(read_promise->get_future(),
                           std::thread(ReadParquetThread,
                                       ctx,
                                       paths[kI],
                                       tableOuts[kI],
                                       read_promise));
    }
    bool all_passed = true;
    for (auto &future : futures) {
      auto status = future.first.get();
      all_passed &= status.is_ok();
      future.second.join();
    }
    return all_passed ? Status::OK() : Status(cylon::IOError, "Failed to read the parquet files");
  } else {
    auto status = Status::OK();
    for (std::size_t kI = 0; kI < paths.size(); ++kI) {
      status = FromParquet(ctx, paths[kI], *tableOuts[kI]);
      if (!status.is_ok()) {
        return status;
      }
    }
    return status;
  }
}
}  // namespace cylon
