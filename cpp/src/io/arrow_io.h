#ifndef TWISTERX_SRC_IO_ARROW_IO_H_
#define TWISTERX_SRC_IO_ARROW_IO_H_

#include <string>
namespace twisterx {
namespace io {

arrow::Status read_csv(const std::string &path, std::shared_ptr<arrow::Table> table);

}
}

#endif //TWISTERX_SRC_IO_ARROW_IO_H_
