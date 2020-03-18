#ifndef TWISTERX_SRC_IO_TABLE_API_H_
#define TWISTERX_SRC_IO_TABLE_API_H_
#include <string>

namespace twisterx {
namespace io {
int read_csv(const std::string &path, const std::string &id);

int column_count(const std::string &id);
}
}
#endif //TWISTERX_SRC_IO_TABLE_API_H_
