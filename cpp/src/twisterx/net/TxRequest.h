//
// Created by vibhatha on 4/15/20.
//

#ifndef TWISTERX_TXREQUEST_H
#define TWISTERX_TXREQUEST_H

#include "iostream"
using namespace std;

namespace twisterx {
class TxRequest {

 public:
  void *buffer{};
  int length{};
  int target;
  int header[6] = {};
  int headerLength{};

  TxRequest(int tgt, void *buf, int len);

  TxRequest(int tgt, void *buf, int len, int *head, int hLength);

  explicit TxRequest(int tgt);

  ~TxRequest();

  void to_string(string dataType, int bufDepth);
};
}

#endif //TWISTERX_TXREQUEST_H
