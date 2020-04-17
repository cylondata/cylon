//
// Created by vibhatha on 4/4/20.
//

#include <map>
#include "table_builder.h"
#include <arrow/api.h>
#include <iostream>
#include "../status.cpp"
#include "../io/arrow_io.hpp"
#include <arrow/io/api.h>
#include <fstream>


using namespace std;

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

twisterx::Status twisterx::data::read_csv() {
    arrow::Result<std::shared_ptr<arrow::Table>> result = static_cast<arrow::Result<std::shared_ptr<arrow::Table>>>(NULL);

    arrow::Status st;
    arrow::MemoryPool *pool = arrow::default_memory_pool();

    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = 10 * sizeof(uint64_t);
    arrow::Status status = AllocateBuffer(arrow::default_memory_pool(), buf_size + 1, &indices_buf);

    for (int64_t i = 0; i < 10; i++) {
        auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
        indices_begin[i] = i;
    }

    //arrow::io::MemoryMappedFile::Open("/tmp/csv.csv", arrow::io::FileMode::READ);
//    string line;
//    ifstream myfile ("/tmp/csv.csv");
//    if (myfile.is_open())
//    {
//        while ( getline (myfile,line) )
//        {
//            cout << line << '\n';
//        }
//        myfile.close();
//    }


//    if (!mmap_result.status().ok()) {
//        mmap_result.status();
//    }

    //arrow::Result<std::shared_ptr<arrow::Table>> result =
    //arrow::Status st;
    //twisterx::io::read_csv(path);
//    if (result.ok()) {
//        std::shared_ptr<arrow::Table> table = *result;
//        put_table(id, table);
//        return twisterx::Status(Code::OK, result.status().message());
//    }
//    twisterx::Status s =  twisterx::Status(Code::IOError, result.status().message());;
    return twisterx::Status::OK();
}