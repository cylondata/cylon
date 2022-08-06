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
#include <arrow/compute/api.h>
#include <arrow/table.h>

#include <fstream>
#include <future>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <algorithm>

#include <cylon/table.hpp>
#include <cylon/arrow/arrow_all_to_all.hpp>
#include <cylon/arrow/arrow_comparator.hpp>
#include <cylon/arrow/arrow_types.hpp>
#include <cylon/ctx/arrow_memory_pool_utils.hpp>
#include <cylon/io/arrow_io.hpp>
#include <cylon/join/join.hpp>
#include <cylon/util/macros.hpp>
#include <cylon/util/arrow_utils.hpp>
#include <cylon/scalar.hpp>
#include <cylon/serialize/table_serialize.hpp>

namespace cylon {
    /**
     * Slice the part of table to create a single table
     * @param table, offset and length
     * @return new sliced table
     */


    Status Slice(const std::shared_ptr<Table> &in, int64_t offset, int64_t length,
                std::shared_ptr<cylon::Table> &out) {
        const auto &ctx = in->GetContext();
    
        std::shared_ptr<arrow::Table> out_table;
        const auto& in_table = in->get_table();

        if (!in->Empty()) {
            out_table = in_table->Slice(offset, length);
        } else {
            out_table = in_table;
        }
        return Table::FromArrowTable(ctx, std::move(out_table), out);
    }


    /**
     * DistributedSlice the part of table to create a single table
     * @param table, offset and length
     * @return new sliced table
     */


    Status DistributedSlice(const std::shared_ptr<cylon::Table> &in, int64_t offset, int64_t length,
                std::shared_ptr<cylon::Table> &out) {

        const auto &ctx = in->GetContext();
        std::shared_ptr<arrow::Table> out_table;
        auto num_row = in->Rows();
        
        std::vector<int64_t> sizes;
        std::shared_ptr<cylon::Column> sizes_cols;
        RETURN_CYLON_STATUS_IF_FAILED(Column::FromVector(sizes, sizes_cols));

        auto num_row_scalar = std::make_shared<Scalar>(arrow::MakeScalar(num_row));
        

        RETURN_CYLON_STATUS_IF_FAILED(ctx->GetCommunicator()->Allgather(num_row_scalar, &sizes_cols));

        auto *data_ptr =
            std::static_pointer_cast<arrow::Int64Array>(sizes_cols->data())
                ->raw_values();

        int64_t L = length;
        int64_t K = offset;
        int64_t zero_0 = 0;
        int64_t rank = ctx->GetRank();
        int64_t L_i = std::accumulate(data_ptr, data_ptr + rank, zero_0);

        int64_t sl_i = *(data_ptr + rank);


        int64_t x = std::max(zero_0, std::min(K - L_i, sl_i));
        int64_t y = std::min(sl_i, std::max(K + L - L_i, zero_0)) - x;


        return Slice(in, x, y, out);
    }


    /**
     * Head the part of table to create a single table with specific number of rows
     * @param tables, number of rows
     * @return new table
     */

    Status Head(const std::shared_ptr<Table> &table, int64_t num_rows, std::shared_ptr<cylon::Table> &output) {

        std::shared_ptr<arrow::Table>  in_table = table->get_table();
        const int64_t table_size = in_table->num_rows();

        if(num_rows > 0 && table_size > 0) {
            return Slice(table, 0, num_rows, output);
        }
        else
            return cylon::Status(Code::IOError, "Number of tailed row should be greater than zero with minimum table elements");
        }

        Status DistributedHead(const std::shared_ptr<Table> &table, int64_t num_rows, std::shared_ptr<cylon::Table> &output) {

        std::shared_ptr<arrow::Table>  in_table = table->get_table();
        const int64_t table_size = in_table->num_rows();

        if(num_rows > 0 && table_size > 0) {
            return DistributedSlice(table, 0, num_rows, output);
        }
        else
            return cylon::Status(Code::IOError, "Number of tailed row should be greater than zero with minimum table elements");

    }

    /**
     * Tail the part of table to create a single table with specific number of rows
     * @param tables, number of rows
     * @return new table
     */

    Status Tail(const std::shared_ptr<Table> &table, int64_t num_rows, std::shared_ptr<cylon::Table> &output) {

        std::shared_ptr<arrow::Table>  in_table = table->get_table();
        const int64_t table_size = in_table->num_rows();

        if(num_rows > 0 && table_size > 0) {
            return Slice(table, table_size-num_rows, num_rows, output);
        }
        else
            return cylon::Status(Code::IOError, "Number of tailed row should be greater than zero with minimum table elements");

        }

    Status DistributedTail(const std::shared_ptr<Table> &table, int64_t num_rows, std::shared_ptr<cylon::Table> &output) {

        std::shared_ptr<arrow::Table>  in_table = table->get_table();
        const int64_t table_size = in_table->num_rows();

        if(num_rows > 0 && table_size > 0) {
            const auto &ctx = table->GetContext();
            std::shared_ptr<arrow::Table> out_table;
            auto num_row = table->Rows();

            std::vector<int64_t> sizes;
            std::shared_ptr<cylon::Column> sizes_cols;
            RETURN_CYLON_STATUS_IF_FAILED(Column::FromVector(sizes, sizes_cols));

            auto num_row_scalar = std::make_shared<Scalar>(arrow::MakeScalar(num_row));


            RETURN_CYLON_STATUS_IF_FAILED(ctx->GetCommunicator()->Allgather(num_row_scalar, &sizes_cols));

            auto *data_ptr =
                std::static_pointer_cast<arrow::Int64Array>(sizes_cols->data())
                    ->raw_values();

            int64_t L = num_rows;
            int64_t zero_0 = 0;
            int64_t rank = ctx->GetRank();
            int64_t L_i = std::accumulate(data_ptr, data_ptr + rank, zero_0);

            int64_t sl_i = *(data_ptr + rank);


            int64_t y = std::max(zero_0, std::min(L - L_i, sl_i));
            int64_t x = sl_i - y;


            return Slice(table, x, y, output);
        }
        else
            return cylon::Status(Code::IOError, "Number of tailed row should be greater than zero with minimum table elements");

    }

}