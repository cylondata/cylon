#ifndef TWISTERX_REQUEST_H
#define TWISTERX_REQUEST_H

#include <memory>

#define TWISTERX_MSG_FIN 1

namespace twisterx {
  /**
   * When a buffer is inserted, we need to return a reference to that buffer
   */
  struct TxRequest {
    void * buffer{};
    int length{};
    int target;
    std::shared_ptr<int> header;
    int headerLength{};

    /**
     * Channel specific holder
     */
    void * channel{};

    TxRequest(int tgt, void *buf, int len, std::shared_ptr<int> head, int hLength) {
      target = tgt;
      buffer = buf;
      length = len;
      header = head;
      headerLength = hLength;
    }

    TxRequest(int tgt) {
      target = tgt;
    }
  };
}

#endif