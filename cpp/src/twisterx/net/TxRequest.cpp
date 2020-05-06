#include "TxRequest.h"
#include <memory>
#include <cstring>
#include "iostream"
#include "../util/builtins.h"

twisterx::TxRequest::TxRequest(int tgt) {
  target = tgt;
}

twisterx::TxRequest::TxRequest(int tgt, void *buf, int len) {
  target = tgt;
  buffer = buf;
  length = len;
}

twisterx::TxRequest::TxRequest(int tgt, void *buf, int len, int *head, int hLength) {
  target = tgt;
  buffer = buf;
  length = len;
  memcpy(&header[0], head, hLength * sizeof(int));
  headerLength = hLength;
}

twisterx::TxRequest::~TxRequest() {
  buffer = nullptr;
}

void twisterx::TxRequest::to_string(string dataType, int bufDepth) {
  std::cout << "Target: " << target << std::endl;
  std::cout << "Length: " << length << std::endl;
  std::cout << "Header Length: " << headerLength << std::endl;
  std::cout << "Buffer: " << std::endl;
  twisterx::util::printArray(buffer, length, dataType, bufDepth);
  std::cout << "Header: " << std::endl;
  for (int i = 0; i < headerLength; ++i) {
    std::cout << header[i] << " ";
  }
  std::cout << std::endl;
}