#ifndef TWISTERX_REQUEST_H
#define TWISTERX_REQUEST_H

#include <memory>
#include <cstring>
#include <glog/logging.h>

#define TWISTERX_MSG_FIN 1

namespace twisterx {
  /**
   * When a buffer is inserted, we need to return a reference to that buffer
   */
  struct TxRequest {
    void * buffer{};
    int length{};
    int target;
    int header[6];
    int headerLength{};

    TxRequest(int tgt, void *buf, int len) {
      target = tgt;
      buffer = buf;
      length = len;
    }

    TxRequest(int tgt, void *buf, int len, int * head, int hLength) {
      target = tgt;
      buffer = buf;
      length = len;
      // we are copying the header
      memcpy(&header[0], head, hLength * sizeof(int));
      headerLength = hLength;
    }

    TxRequest(int tgt) {
      target = tgt;
    }

    ~TxRequest() {
      // LOG(INFO) << "Delete the request with address" << buffer;
      buffer = nullptr;
    };
  };
}

#endif