#ifndef TWISTERX_BUFFER_H
#define TWISTERX_BUFFER_H

#include <cstring>

namespace twister::comm {

    typedef unsigned char byte;

    class Buffer {
    private:
        byte *buff;
        int limit;
        int index = 0;

    public:
        Buffer(int &size) {
            this->buff = new byte[size];
            for (int i = 0; i < 100; i++) {
                this->buff[i] = 0;
            }
            this->limit = size;
        }

        ~Buffer() {
            this->clear();
        }

        bool put_int(int val) {
            if (index + sizeof(val) < this->limit) {
                memcpy(this->buff + index, &val, sizeof(val));
                index += sizeof(val);
                return true;
            }
            return false;
        }

        int get_int() {
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
