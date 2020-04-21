#ifndef TWISTERX_TABLE_BUILDER_H
#define TWISTERX_TABLE_BUILDER_H
#include "string"
#include <arrow/api.h>
#include "../io/arrow_io.hpp"
#include "../join/join.hpp"
#include "../status.hpp"

using namespace std;

namespace twisterx {
namespace data {

void read_csv();

int get_rows();

int get_columns();

string get_id();

}
}

#endif TWISTERX_TABLE_BUILDER_H
