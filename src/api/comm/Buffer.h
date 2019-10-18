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
            if (index + sizeof(val) < this->limit) {
                memcpy(this->buff + index, &val, sizeof(val));
                index += sizeof(val);
                return true;
            }
            return false;
        }

        int get_int32() {
            if (index + sizeof(int) < this->limit) {
                int val = 0;
                memcpy(&val, this->buff + index, sizeof(val));
                index += sizeof(val);
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
