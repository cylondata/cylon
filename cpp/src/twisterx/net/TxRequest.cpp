//
// Created by vibhatha on 4/15/20.
//

#include "TxRequest.h"
#include <memory>
#include <cstring>
#include "iostream"

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

void twisterx::TxRequest::to_string() {
    std::cout << "Target: " << target << std::endl;
    std::cout << "Length: " << length << std::endl;
    std::cout << "Header Length: " << headerLength << std::endl;
    std::cout << "Buffer: " << std::endl;
    for (int i = 0; i < length; ++i) {
        std::cout << ((int32_t *)buffer)[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Header: " << std::endl;
    for (int i = 0; i < headerLength; ++i) {
        std::cout << header[i] << " ";
    }
    std::cout << std::endl;
}