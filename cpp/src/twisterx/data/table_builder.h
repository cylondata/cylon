//
// Created by vibhatha on 4/4/20.
//

#ifndef TWISTERX_TABLE_BUILDER_H
#define TWISTERX_TABLE_BUILDER_H

#include "string"
#include "../status.cpp"
#include <map>
#include <arrow/api.h>
#include "../status.cpp"
#include "../io/arrow_io.hpp"


using namespace std;

namespace twisterx {
    namespace data {

        twisterx::Status read_csv();//(const std::string &path, const std::string &id);

        int get_rows();

        int get_columns();

        string get_id();

    }
}


#endif //TWISTERX_TABLE_BUILDER_H
