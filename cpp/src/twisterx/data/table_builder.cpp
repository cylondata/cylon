//
// Created by vibhatha on 4/4/20.
//

#include <map>
#include "table_builder.h"
#include <arrow/api.h>
#include "../status.cpp"
#include "../io/arrow_io.hpp"

string twisterx::data::get_id() {
    return "table1";
}

int twisterx::data::get_rows() {
    return 20;
}

int twisterx::data::get_columns() {
    return 10;
}

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

twisterx::Status twisterx::data::read_csv(const std::string &path, const std::string &id) {
    arrow::Result<std::shared_ptr<arrow::Table>> result = twisterx::io::read_csv(path);
    return twisterx::Status::OK();
}