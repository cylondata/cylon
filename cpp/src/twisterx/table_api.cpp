#include "table_api.hpp"
#include <memory>
#include <arrow/api.h>
#include <map>
#include "io/arrow_io.hpp"
#include "join/join.hpp"
#include  "util/to_string.hpp"
#include "iostream"
#include <glog/logging.h>
#include "util/arrow_utils.hpp"
#include "arrow/arrow_partition_kernels.hpp"

namespace twisterx {

    std::map<std::string, std::shared_ptr<arrow::Table>> table_map{}; //todo make this un ordered

    std::shared_ptr<arrow::Table> get_table(const std::string &id) {
        auto itr = table_map.find(id);
        if (itr != table_map.end()) {
            return itr->second;
        }
        return NULLPTR;
    }

    void put_table(const std::string &id, const std::shared_ptr<arrow::Table> &table) {
        std::pair<std::string, std::shared_ptr<arrow::Table>> pair(id, table);
        table_map.insert(pair);
    }

    twisterx::Status read_csv(const std::string &path,
                              const std::string &id,
                              twisterx::io::config::CSVReadOptions options) {
        arrow::Result<std::shared_ptr<arrow::Table>> result = twisterx::io::read_csv(path, options);
        if (result.ok()) {
            std::shared_ptr<arrow::Table> table = *result;
            put_table(id, table);
            return twisterx::Status(Code::OK, result.status().message());
        }
        return twisterx::Status(Code::IOError, result.status().message());;
    }

    twisterx::Status print(const std::string &table_id, int col1, int col2, int row1, int row2) {
        return print_to_ostream(table_id, col1, col2, row1, row2, std::cout);
    }

    twisterx::Status print_to_ostream(const std::string &table_id,
                                      int col1,
                                      int col2,
                                      int row1,
                                      int row2,
                                      std::ostream &out) {
        auto table = get_table(table_id);
        if (table != NULLPTR) {
            for (int row = row1; row < row2; row++) {
                for (int col = col1; col < col2; col++) {
                    auto column = table->column(col);
                    int rowCount = 0;
                    for (int chunk = 0; chunk < column->num_chunks(); chunk++) {
                        auto array = column->chunk(chunk);
                        if (rowCount <= row && rowCount + array->length() > row) {
                            // print this array
                            out << twisterx::util::array_to_string(array, row - rowCount);
                            if (col != col2 - 1) {

                                out << ",";
                            }
                            break;
                        }
                        rowCount += array->length();
                    }
                }
                out << std::endl;
            }
        }
        return twisterx::Status(Code::OK);
    }

    twisterx::Status JoinTables(const std::string &table_left,
                                const std::string &table_right,
                                twisterx::join::config::JoinConfig join_config,
                                const std::string &dest_id) {
        auto left = get_table(table_left);
        auto right = get_table(table_right);

        if (left == NULLPTR) {
            return twisterx::Status(Code::KeyError, "Couldn't find the left table");
        } else if (right == NULLPTR) {
            return twisterx::Status(Code::KeyError, "Couldn't find the right table");
        } else {
            std::shared_ptr<arrow::Table> table;
            arrow::Status status = join::joinTables(
                    left,
                    right,
                    join_config,
                    &table,
                    arrow::default_memory_pool()
            );
            put_table(dest_id, table);
            return twisterx::Status((int) status.code(), status.message());
        }
    }

    int column_count(const std::string &id) {
        auto table = get_table(id);
        if (table != NULLPTR) {
            return table->num_columns();
        }
        return -1;
    }

    int row_count(const std::string &id) {
        auto table = get_table(id);
        if (table != NULLPTR) {
            return table->num_rows();
        }
        return -1;
    }

    twisterx::Status merge(std::vector<std::string> table_ids, const std::string &merged_tab) {
        std::vector<std::shared_ptr<arrow::Table>> tables;
        for (auto it = table_ids.begin(); it < table_ids.end(); it++) {
            tables.push_back(get_table(*it));
        }
        arrow::Result<std::shared_ptr<arrow::Table>> result = arrow::ConcatenateTables(tables);
        if (result.status() == arrow::Status::OK()) {
            put_table(merged_tab, result.ValueOrDie());
            return twisterx::Status::OK();
        } else {
            return twisterx::Status((int) result.status().code(), result.status().message());
        }
    }

    twisterx::Status sortTable(const std::string &id, const std::string &sortedTableId, int columnIndex) {
        auto table = get_table(id);
        if (table == NULLPTR) {
            LOG(FATAL) << "Failed to retrieve table";
            return Status(Code::KeyError, "Couldn't find the right table");
        }
        auto col = table->column(columnIndex)->chunk(0);
        std::shared_ptr<arrow::Array> indexSorts;
        arrow::Status status = SortIndices(arrow::default_memory_pool(), col, &indexSorts);

        if (status != arrow::Status::OK()) {
            LOG(FATAL) << "Failed when sorting table to indices. " << status.ToString();
            return twisterx::Status((int) status.code(), status.message());
        }

        std::vector<std::shared_ptr<arrow::Array>> data_arrays;
        for (auto &column : table->columns()) {
            std::shared_ptr<arrow::Array> destination_col_array;
            status = twisterx::util::copy_array_by_indices(nullptr, column->chunk(0),
                                                           &destination_col_array, arrow::default_memory_pool());
            if (status != arrow::Status::OK()) {
                LOG(FATAL) << "Failed while copying a column to the final table from left table. " << status.ToString();
                return twisterx::Status((int) status.code(), status.message());
            }
            data_arrays.push_back(destination_col_array);
        }
        // we need to put this to a new place
        std::shared_ptr<arrow::Table> sortedTable = arrow::Table::Make(table->schema(), data_arrays);
        put_table(sortedTableId, sortedTable);
        return Status::OK();
    }

    twisterx::Status hashPartition(const std::string &id, const std::vector<int> &hash_columns, int no_of_partitions,
                                   std::vector<std::shared_ptr<arrow::Table>> *out, arrow::MemoryPool *pool) {
        std::shared_ptr<arrow::Table> left_tab = get_table(id);
        // keep arrays for each target, these arrays are used for creating the table
        std::unordered_map<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;
        std::vector<int> partitions;
        for (int t = 0; t < no_of_partitions; t++) {
            partitions.push_back(t);
            data_arrays.insert(
                    std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(
                            t, std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>()));
        }

        std::vector<std::shared_ptr<arrow::Array>> arrays;
        int64_t length = 0;
        for (auto col_index: hash_columns) {
            auto column = left_tab->column(col_index);
            std::vector<int64_t> outPartitions;
            std::shared_ptr<arrow::Array> array = column->chunk(0);
            arrays.push_back(array);

            if (!(length == 0 || length == column->length())) {
                return twisterx::Status(twisterx::IndexError, "Column lengths doesnt match " + std::to_string(length));
            }
            length = column->length();
        }

        // first we partition the table
        std::vector<int64_t> outPartitions;
        twisterx::Status status = HashPartitionArrays(pool, arrays, length, partitions, &outPartitions);
        if (!status.is_ok()) {
            LOG(FATAL) << "Failed to create the hash partition";
            return status;
        }

        for (int i = 0; i < left_tab->num_columns(); i++) {
            std::shared_ptr<arrow::DataType> type = left_tab->column(i)->chunk(0)->type();
            std::shared_ptr<arrow::Array> array = left_tab->column(i)->chunk(0);

            std::unique_ptr<ArrowArraySplitKernel> splitKernel;
            status = CreateSplitter(type, pool, &splitKernel);
            if (!status.is_ok()) {
                LOG(FATAL) << "Failed to create the splitter";
                return status;
            }

            // this one outputs arrays for each target as a map
            std::unordered_map<int, std::shared_ptr<arrow::Array>> splited_arrays;
            splitKernel->Split(array, outPartitions, partitions, splited_arrays);

            for (const auto &x : splited_arrays) {
                std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>> cols = data_arrays[x.first];
                cols->push_back(x.second);
            }
        }
        // now insert these array to
        for (const auto &x : data_arrays) {
            std::shared_ptr<arrow::Table> table = arrow::Table::Make(left_tab->schema(), *x.second);
            out->push_back(table);
        }
        return twisterx::Status::OK();
    }
    //TODO: Move to python dir within cpp
    twisterx::Status from_csv(const std::string &path, const std::string &id, const char delimiter) {
        read_csv(path, id, twisterx::io::config::CSVReadOptions().WithDelimiter(delimiter));
    }

}