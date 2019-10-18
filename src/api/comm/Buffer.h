#ifndef TWISTERX_BUFFER_H
#define TWISTERX_BUFFER_H

#include <cstring>
#include <stdexcept>

#define data_type(name, type)\
    bool put_##name(type val) { \
        if (ensure_capacity(sizeof(type))) { \
            memcpy(this->buff + index, &val, sizeof(type)); \
            index += sizeof(type); \
            return true; \
        } \
        return false; \
    } \
    \
    type get_##name() { \
        if (ensure_capacity(sizeof(type))) { \
            type val = 0; \
            memcpy(&val, this->buff + index, sizeof(type)); \
            index += sizeof(type); \
            return val; \
        } else { \
            throw std::runtime_error("Can't read an int from this buffer"); \
        } \
    }


namespace twisterx::comm {

    typedef unsigned char byte;

    class Buffer {
    private:
        byte *buff;
        int32_t limit;
        int32_t index = 0;

    public:
        explicit Buffer(int32_t &size) {
            this->buff = new byte[size];
            for (int i = 0; i < size; i++) {
                this->buff[i] = 0;
            }
            this->limit = size;
        }

        ~Buffer() {
            this->clear();
        }

        bool ensure_capacity(size_t size) {
            return index + size < this->limit;
        }

        int32_t remaining() {
            return limit - index;
        }

        bool is_full() {
            return this->remaining() == 0;
        }

        data_type(bool, bool)

        data_type(char, char)

        data_type(int8, int8_t);

        data_type(int32, int32_t)

        data_type(int64, int64_t)

        data_type(float, float)

        data_type(double, double)

        void flip() {
            this->index = 0;
        }

        void clear() {
            delete this->buff;
            this->index = 0;
        }
    };
}

#undef data_type

#endif //TWISTERX_BUFFER_H
