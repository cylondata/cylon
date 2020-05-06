#ifndef TWISTERX_SRC_TWISTERX_IO_CSV_READ_CONFIG_HOLDER_HPP_
#define TWISTERX_SRC_TWISTERX_IO_CSV_READ_CONFIG_HOLDER_HPP_
#include <arrow/csv/options.h>
#include "csv_read_config.h"

namespace twisterx {
namespace io {
namespace config {
/**
 * This is a helper class to hold the arrow CSV read options.
 * This class shouldn't be used with other language interfaces, due to the arrow
 * dependency.
 */
class CSVConfigHolder : public arrow::csv::ReadOptions,
						public arrow::csv::ParseOptions,
						public arrow::csv::ConvertOptions {
 public:
  static CSVConfigHolder *GetCastedHolder(const CSVReadOptions &options) {
	void *holder = options.GetHolder();
	return (CSVConfigHolder *)(holder);
  }
};
}
}
}
#endif //TWISTERX_SRC_TWISTERX_IO_CSV_READ_CONFIG_HOLDER_HPP_
