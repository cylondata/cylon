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
#include <gcylon/gtable_api.hpp>
#include <cylon/util/macros.hpp>

namespace gcylon {

/**
 * create a table with empty columns
 * each column has the same datatype with the given table column
 * @param tv
 * @return
 */
std::unique_ptr<cudf::table> createEmptyTable(const cudf::table_view & tv) {

    std::vector<std::unique_ptr<cudf::column>> columnVector{};
    for (int i=0; i < tv.num_columns(); i++) {
        auto column = std::make_unique<cudf::column>(tv.column(i).type(),
                                                     0,
                                                     rmm::device_buffer{0, rmm::cuda_stream_default});
        columnVector.push_back(std::move(column));
    }

    return std::make_unique<cudf::table>(std::move(columnVector));
}

cylon::Status all_to_all_cudf_table(std::shared_ptr<cylon::CylonContext> ctx,
                                    std::unique_ptr<cudf::table> & ptable,
                                    std::vector<cudf::size_type> &offsets,
                                    std::unique_ptr<cudf::table> &table_out) {

    const auto &neighbours = ctx->GetNeighbours(true);
    std::vector<std::shared_ptr<cudf::table>> received_tables;
    received_tables.reserve(neighbours.size());

    // define call back to catch the receiving tables
    CudfCallback cudf_callback =
            [&received_tables](int source, const std::shared_ptr<cudf::table> &table_, int reference) {
                received_tables.push_back(table_);
                return true;
            };

    // doing all to all communication to exchange tables
    CudfAllToAll all_to_all(ctx, neighbours, neighbours, ctx->GetNextSequence(), cudf_callback);

    // insert partitioned table for all-to-all
    cudf::table_view tv = ptable->view();
    int accepted = all_to_all.insert(tv, offsets, ctx->GetNextSequence());
    if (!accepted)
        return cylon::Status(accepted);

    // wait for the partitioned tables to arrive
    // now complete the communication
    all_to_all.finish();
    while (!all_to_all.isComplete()) {}
    all_to_all.close();

    if (received_tables.size() == 0) {
        table_out = std::move(createEmptyTable(ptable->view()));
        return cylon::Status::OK();
    }

    std::vector<cudf::table_view> tables_to_concat{};
    for (auto t : received_tables) {
        tables_to_concat.push_back(t->view());
    }

    std::unique_ptr<cudf::table> concatTable = cudf::concatenate(tables_to_concat);

    table_out = std::move(concatTable);
    return cylon::Status::OK();
}

cylon::Status Shuffle(const cudf::table_view & inputTable,
                      const std::vector<int> &columns_to_hash,
                      std::shared_ptr<cylon::CylonContext> ctx,
                      std::unique_ptr<cudf::table> &table_out) {

    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> partitioned
            = cudf::hash_partition(inputTable, columns_to_hash, ctx->GetWorldSize());

    RETURN_CYLON_STATUS_IF_FAILED(
            all_to_all_cudf_table(ctx, partitioned.first, partitioned.second, table_out));

    return cylon::Status::OK();
}

cylon::Status Shuffle(std::shared_ptr<GTable> &inputTable,
                      const std::vector<int> &columns_to_hash,
                      std::shared_ptr<GTable> &outputTable) {

    std::unique_ptr<cudf::table> table_out;
    auto ctx = inputTable->GetContext();

    RETURN_CYLON_STATUS_IF_FAILED(
            Shuffle(inputTable->GetCudfTable()->view(), columns_to_hash, ctx, table_out));

    RETURN_CYLON_STATUS_IF_FAILED(
            GTable::FromCudfTable(ctx, table_out, outputTable));

    // set metadata for the shuffled table
    outputTable->SetCudfMetadata(inputTable->GetCudfMetadata());

    return cylon::Status::OK();
}

cylon::Status joinTables(const cudf::table_view & left,
                         const cudf::table_view & right,
                         const cylon::join::config::JoinConfig &join_config,
                         std::shared_ptr<cylon::CylonContext> ctx,
                         std::unique_ptr<cudf::table> &table_out) {

    if(join_config.GetAlgorithm() == cylon::join::config::JoinAlgorithm::SORT) {
        return cylon::Status(cylon::Code::NotImplemented, "SORT join is not supported on GPUs yet.");
    }

    if(join_config.GetType() == cylon::join::config::JoinType::INNER) {
       table_out = cudf::inner_join(left,
                                    right,
                                    join_config.GetLeftColumnIdx(),
                                    join_config.GetRightColumnIdx());
    } else if (join_config.GetType() == cylon::join::config::JoinType::LEFT) {
        table_out = cudf::left_join(left,
                                    right,
                                    join_config.GetLeftColumnIdx(),
                                    join_config.GetRightColumnIdx());
    } else if (join_config.GetType() == cylon::join::config::JoinType::RIGHT) {
        table_out = cudf::left_join(right,
                                    left,
                                    join_config.GetRightColumnIdx(),
                                    join_config.GetLeftColumnIdx());
    } else if (join_config.GetType() == cylon::join::config::JoinType::FULL_OUTER) {
        table_out = cudf::full_join(left,
                                    right,
                                    join_config.GetLeftColumnIdx(),
                                    join_config.GetRightColumnIdx());
    }

    return cylon::Status::OK();
}


cylon::Status joinTables(std::shared_ptr<GTable> &left,
                         std::shared_ptr<GTable> &right,
                         const cylon::join::config::JoinConfig &join_config,
                         std::shared_ptr<GTable> &joinedTable) {

    if (left == nullptr) {
        return cylon::Status(cylon::Code::KeyError, "Couldn't find the left table");
    } else if (right == nullptr) {
        return cylon::Status(cylon::Code::KeyError, "Couldn't find the right table");
    }

    if(join_config.GetAlgorithm() == cylon::join::config::JoinAlgorithm::SORT) {
        return cylon::Status(cylon::Code::NotImplemented, "SORT join is not supported on GPUs yet.");
    }

    std::shared_ptr<cylon::CylonContext> ctx = left->GetContext();
    std::unique_ptr<cudf::table> joined;
    RETURN_CYLON_STATUS_IF_FAILED(joinTables(left->GetCudfTable()->view(),
                                             right->GetCudfTable()->view(),
                                             join_config,
                                             ctx,
                                             joined));

    RETURN_CYLON_STATUS_IF_FAILED(
            GTable::FromCudfTable(ctx, joined, joinedTable));

    // set metadata for the joined table
    joinedTable->SetCudfMetadata(left->GetCudfMetadata());
    return cylon::Status::OK();
}

 /**
 * Similar to local join, but performs the join in a distributed fashion
  * works on tale_view objects
 * @param leftTable
 * @param rightTable
 * @param join_config
 * @param ctx
 * @param table_out
 * @return
 */
cylon::Status DistributedJoin(const cudf::table_view & leftTable,
                              const cudf::table_view & rightTable,
                              const cylon::join::config::JoinConfig &join_config,
                              std::shared_ptr<cylon::CylonContext> ctx,
                              std::unique_ptr<cudf::table> &table_out) {

    if (ctx->GetWorldSize() == 1) {
        // perform single join
        return joinTables(leftTable, rightTable, join_config, ctx, table_out);
    }

    std::unique_ptr<cudf::table> left_shuffled_table, right_shuffled_table;

    RETURN_CYLON_STATUS_IF_FAILED(
            Shuffle(leftTable, join_config.GetLeftColumnIdx(), ctx, left_shuffled_table));

    RETURN_CYLON_STATUS_IF_FAILED(
            Shuffle(rightTable, join_config.GetRightColumnIdx(), ctx, right_shuffled_table));

    RETURN_CYLON_STATUS_IF_FAILED(
            joinTables(left_shuffled_table->view(), right_shuffled_table->view(), join_config, ctx, table_out));

    return cylon::Status::OK();
}

/**
 * Similar to local join, but performs the join in a distributed fashion
 * works on GTable objects
 * @param left
 * @param right
 * @param join_config
 * @param output
 * @return
 */
cylon::Status DistributedJoin(std::shared_ptr<GTable> &left,
                              std::shared_ptr<GTable> &right,
                              const cylon::join::config::JoinConfig &join_config,
                              std::shared_ptr<GTable> &output) {

    std::shared_ptr<cylon::CylonContext> ctx = left->GetContext();
    if (ctx->GetWorldSize() == 1) {
        // perform single join
        return joinTables(left, right, join_config, output);
    }

    std::shared_ptr<GTable> left_shuffled_table, right_shuffled_table;

    RETURN_CYLON_STATUS_IF_FAILED(
            Shuffle(left, join_config.GetLeftColumnIdx(), left_shuffled_table));

    RETURN_CYLON_STATUS_IF_FAILED(
            Shuffle(right, join_config.GetRightColumnIdx(), right_shuffled_table));

    RETURN_CYLON_STATUS_IF_FAILED(
            joinTables(left_shuffled_table, right_shuffled_table, join_config, output));

    return cylon::Status::OK();
}

/**
 * write GTable to file
 * @param table
 * @param outputFile
 * @return
 */
cylon::Status WriteToCsv(std::shared_ptr<GTable> &table, std::string outputFile) {
    cudf::io::sink_info sink_info(outputFile);
    cudf::io::csv_writer_options options =
            cudf::io::csv_writer_options::builder(sink_info, table->GetCudfTable()->view())
            .metadata(&(table->GetCudfMetadata()))
            .include_header(true);
    cudf::io::write_csv(options);
    return cylon::Status::OK();
}

}// end of namespace gcylon
