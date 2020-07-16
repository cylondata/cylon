#include "arrow_builder.hpp"
#include "../table_api_extended.hpp"
#include <arrow/ipc/reader.h>
#include <iostream>
#include <glog/logging.h>

void cylon::carrow::Build(std::string table_id,
                          uint8_t *schema,
                          int64_t schema_length,
                          std::vector<int8_t *> buffers,
                          std::vector<int64_t> lengths) {

  LOG(INFO) << "inside build functions" << schema_length;

  for (int32_t i = 0; i < schema_length; i++) {
    std::cout << schema[i] << ",";
  }
  std::cout << std::endl;

  std::unique_ptr<arrow::ipc::Message> msg;
  auto buffer = std::make_shared<arrow::Buffer>(schema, schema_length);
  LOG(INFO) << "before reading header";
  auto status = arrow::ipc::Message::Open(buffer, buffer, &msg);
  arrow::ipc::DictionaryMemo dictionary_memo;
  std::shared_ptr<arrow::Schema> schema_out;

  //arrow::ipc::Message msgp(nullptr, nullptr);

  status = arrow::ipc::ReadSchema(*msg, &dictionary_memo, &schema_out);

  if (status.ok()) {
    for (const auto &f: schema_out->fields()) {
      LOG(INFO) << "Field : " << f->name();
    }
  }
}
