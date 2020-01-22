#ifndef TWISTERX_CALLBACK_H
#define TWISTERX_CALLBACK_H

class ReceiveCallback {
public:
  virtual bool onReceive(int source, void * buffer, int length) = 0;
};

#endif