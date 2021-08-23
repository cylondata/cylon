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

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <cudf/partitioning.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/join.hpp>
#include <cudf/io/csv.hpp>

#include <gcylon/gtable.hpp>
#include <cylon/util/macros.hpp>

namespace gcylon {

GTable::GTable(std::shared_ptr<cylon::CylonContext> &ctx, std::unique_ptr<cudf::table> &tab)
    : id_("0"), ctx_(ctx), table_(std::move(tab)) {}

GTable::GTable(std::shared_ptr<cylon::CylonContext> &ctx,
               std::unique_ptr<cudf::table> &tab,
               cudf::io::table_metadata &metadata)
            : id_("0"), ctx_(ctx), table_(std::move(tab)), metadata_(metadata) {}

GTable::~GTable() {}

std::shared_ptr<cylon::CylonContext> GTable::GetContext() {
    return this->ctx_;
}

std::unique_ptr<cudf::table> & GTable::GetCudfTable() {
    return this->table_;
}

cudf::io::table_metadata & GTable::GetCudfMetadata() {
    return this->metadata_;
}

    /**
     * sets cudf table metadata
     * @return
     */
void GTable::SetCudfMetadata(cudf::io::table_metadata & metadata) {
    metadata_ = metadata;
}

cylon::Status GTable::FromCudfTable(std::shared_ptr<cylon::CylonContext> &ctx,
                                    std::unique_ptr<cudf::table> &table,
                                    std::shared_ptr<GTable> &tableOut) {
    if (false) { // todo: need to check column types
        LOG(FATAL) << "Types not supported";
        return cylon::Status(cylon::Invalid, "This type not supported");
    }
    tableOut = std::make_shared<GTable>(ctx, table);
    return cylon::Status(cylon::OK, "Loaded Successfully");
}

cylon::Status GTable::FromCudfTable(std::shared_ptr<cylon::CylonContext> &ctx,
                                    cudf::io::table_with_metadata &table,
                                    std::shared_ptr<GTable> &tableOut) {
    if (false) { // todo: need to check column types
        LOG(FATAL) << "Types not supported";
        return cylon::Status(cylon::Invalid, "This type not supported");
    }

    tableOut = std::make_shared<GTable>(ctx, table.tbl, table.metadata);
    return cylon::Status(cylon::OK, "Loaded Successfully");
}

}// end of namespace gcylon
