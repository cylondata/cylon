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

        Buffer() = delete;

    public:
        explicit Buffer(int &size) {
            this->buff = new byte[size];
            this->limit = size;
        }

        bool put_int(int val) {
            if (index + 4 < this->limit) {
                memcpy(this->buff + index, &val, 4);
                index += 4;
                return true;
            }
            return false;
        }

        int get_int() {
            int val = 0;
            memcpy(&val, this->buff + index, 4);
            index += 4;
            return val;
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
