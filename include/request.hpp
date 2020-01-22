#ifndef TWISTERX_REQUEST_H
#define TWISTERX_REQUEST_H

namespace twisterx {
  /**
   * When a buffer is inserted, we need to return a reference to that buffer
   */
  struct TxRequest {
    void * buffer;
    int length;
    int target;

    /**
     * Channel specific holder
     */
    void * channel{};

    TxRequest(int tgt, void *buf, int len) {
      target = tgt;
      buffer = buf;
      length = len;
    }
  };
}

#endif