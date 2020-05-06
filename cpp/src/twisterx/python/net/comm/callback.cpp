#include "callback.h"

bool twisterx::net::comms::Callback::onReceive(int source, void *buffer, int length) {
  std::cout << "Received value: " << source << " length " << length << std::endl;
  delete[] reinterpret_cast<char *>(buffer);
  return false;
}

bool twisterx::net::comms::Callback::onReceiveHeader(int source, int finished, int *buffer, int length) {
  std::cout << "Received HEADER: " << source << " length " << length << std::endl;
  return false;
}

bool twisterx::net::comms::Callback::onSendComplete(int target, void *buffer, int length) {
  return false;
}