#ifndef __SMALL_TWISTER_H_
#define __SMALL_TWISTER_H_
#include <string>
#include <type_traits>
#include <string.h>

template<typename T>
struct is_char_pointer { static const bool value = false; };

template<>
struct is_char_pointer<char*> { static const bool value = true; };

template<>
struct is_char_pointer<std::string> { static const bool value = true; };

template<>
struct is_char_pointer<const char*> { static const bool value = true; };

template<typename... Ts>
class record_t {

public:
  // we may need to pass join field indexes as an array ...
  record_t(unsigned char* const _buffer, uint32_t* _pStart, Ts ... vals) : buffer(_buffer), start(_pStart)  {
    r_size = size(vals...); 
    serialize(vals...);
  }

  uint32_t get_record_size() {
    return r_size;
  }

  uint32_t get_buffer_start() {
    return (*start);
  }

  uint32_t get_rank() {
    // calculate rank based on join fields and a hash function ...
    return 0;
  }


private:
  template<typename T>
  void serialize(T v) {
    memcpy((buffer+(*start)), &v, sizeof(v));
    (*start) += sizeof(v);
  }

  template<typename T, typename... Args>
  void serialize(T first, Args ... rest) {
  }

  // Do same for serialize ...
  template<typename T>
  typename std::enable_if<!is_char_pointer<T>::value, uint32_t>::type size(T v) {
    return sizeof(v);
  }


  uint32_t size(char* str) {
    return sizeof(uint16_t) + strlen(str);
  }

  uint32_t size(const char* str) {
    return sizeof(uint16_t) + strlen(str);
  }

  uint32_t size(std::string str) {
    return size(str.c_str());
  }

  template<typename T, typename... Args>
  uint32_t size(T first, Args ... rest) {
    return (size(first) + size(rest...));
  }

  const unsigned char* buffer;
  uint32_t* start;
  uint32_t r_size;

};

#endif
