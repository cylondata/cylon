#ifndef TWISTERX_BUFFER_H
#define TWISTERX_BUFFER_H

#include <cstring>

namespace twisterx::comm {

    typedef unsigned char byte;

    class Buffer {
    private:
        byte *buff;
        int32_t limit;
        int32_t index = 0;

        bool ensure_capacity(uint16_t size) {
            return index + size < this->limit;
        }

    public:
        Buffer(int32_t &size) {
            this->buff = new byte[size];
            for (int i = 0; i < size; i++) {
                this->buff[i] = 0;
            }
            this->limit = size;
        }

        ~Buffer() {
            this->clear();
        }

        bool put_int32(int32_t val) {
            if (ensure_capacity(sizeof(int32_t))) {
                memcpy(this->buff + index, &val, sizeof(int32_t));
                index += sizeof(int32_t);
                return true;
            }
            return false;
        }

        int32_t get_int32() {
            if (ensure_capacity(sizeof(int32_t))) {
                int32_t val = 0;
                memcpy(&val, this->buff + index, sizeof(int32_t));
                index += sizeof(int32_t);
                return val;
            } else {
                throw std::runtime_error("Can't read an int from this buffer");
            }
        }

        void flip() {
            this->index = 0;
        }

        void clear() {
            delete this->buff;
            this->index = 0;
        }
    };
}

#endif //TWISTERX_BUFFER_H
