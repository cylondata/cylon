#ifndef TWISTERX_SRC_IO_ARROW_IO_H_
#define TWISTERX_SRC_IO_ARROW_IO_H_

#include <string>
#include "csv_read_config.h"
namespace twisterx {
namespace io {

arrow::Result<std::shared_ptr<arrow::Table>> read_csv(const std::string &path,
													  twisterx::io::config::CSVReadOptions options = twisterx::io::config::CSVReadOptions());

}
}

#endif //TWISTERX_SRC_IO_ARROW_IO_H_
