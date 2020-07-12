#include "arrow_builder.hpp"
#include "../table_api_extended.hpp"
#include <arrow/ipc/reader.h>

void cylon::carrow::Build(std::string table_id,
                          uint8_t *schema,
                          int64_t schema_length,
                          std::vector<int8_t *> buffers,
                          std::vector<int64_t> lengths) {
  std::unique_ptr<arrow::ipc::Message> msg;
  auto buffer = std::make_shared<arrow::Buffer>(schema, schema_length);
  auto status = arrow::ipc::Message::Open(buffer, nullptr, &msg);
  arrow::ipc::DictionaryMemo dictionary_memo;
  std::shared_ptr<arrow::Schema> schema_out;

  //arrow::ipc::Message msgp(nullptr, nullptr);

  arrow::ipc::ReadSchema(*msg, &dictionary_memo, &schema_out);
}
