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
        size_t limit;
        size_t index = 0;

        Buffer() = delete;

    public:
        explicit Buffer(size_t &size) {
            this->buff = new byte[size];
            for (size_t i = 0; i < size; i++) {
                this->buff[i] = 0;
            }
            this->limit = size;
        }

        ~Buffer() {
            this->clear();
            delete[] this->buff;
        }

        Buffer(const Buffer &) = delete;

        bool ensure_capacity(size_t size) {
            return index + size <= this->limit;
        }

        size_t remaining() {
            return limit - index;
        }

        size_t size() {
            return this->limit;
        }

        bool is_full() {
            return this->remaining() == 0;
        }

        size_t position() {
            return this->index;
        }

        data_type(bool, bool)

        data_type(char, char)

        data_type(int8, int8_t);

        data_type(int32, int32_t)

        data_type(int64, int64_t)

        data_type(float, float)

        data_type(double, double)

        bool put(void *data, size_t bytes_count) {
            if (ensure_capacity(bytes_count)) {
                memcpy(this->buff + index, data, bytes_count);
                index += bytes_count;
                return true;
            }
            return false;
        }

        bool get(void *dest, size_t bytes_count) {
            if (ensure_capacity(bytes_count)) {
                memcpy(dest, this->buff + index, bytes_count);
                index += bytes_count;
            } else {
                throw std::runtime_error("Can't read " + std::to_string(bytes_count) + " bytes from this buffer");
            }
        }

        void flip() {
            this->index = 0;
        }

        void clear() {
            // todo reset buffer??
            this->index = 0;
        }

        byte *get_buffer() {
            return this->buff;
        }
    };
}

#undef data_type

#endif //TWISTERX_BUFFER_H
