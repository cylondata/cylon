#ifndef PACK_STRING_LIB_H
#define PACK_STRING_LIB_H

#include <iostream>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

/**
 * @brief Macro allows realloc to be used in CPP with cast requirements
 */
#define CPP_REALLOC(ptr, type, new_size) reinterpret_cast<type*>(std::realloc(reinterpret_cast<void*>(ptr), new_size))

/**
 * @brief Small exception class to handle array related errors for -s MallocArray
 *        
 */


namespace cylon {
    namespace util {
        struct MallocArrayException : public std::exception {
            enum ErrType {
                NULL_ON_MALLOC,
                NULL_ON_REALLOC,
                OUT_OF_BOUNDS,
                SHRINK_LARGER
            };

            MallocArrayException(MallocArrayException::ErrType code);

            const char *what() const throw() {
                return message;
            }

            const char *message;
            ErrType code;
        };

/**
 * @class MallocArray
 * @brief A template class array that uses memory from malloc and realloc
 * @detail This is a class meant to enhance performance by using a template array
 *         that derives memory from the C standard library functions.
 *         In particular, C stdlib has realloc, while C++ does not. If there is
 *         a chance to extend existing memory without a copy, it's a huge perf
 *         boost.
 */
        template<class T>
        class MallocArray {
        public:
            MallocArray(size_t size = 20) : _len(0),
                                            _cap(size * sizeof(T)),
                                            _items(reinterpret_cast<T *>(std::malloc(_cap))) {
                if (_items == nullptr) throw MallocArrayException(MallocArrayException::NULL_ON_MALLOC);
            }

            MallocArray(const MallocArray &other) : _len(other.getLen()),
                                                    _cap(other.getCap()),
                                                    _items(reinterpret_cast<T *>(std::malloc(_cap))) {
                if (_items == nullptr) throw MallocArrayException(MallocArrayException::NULL_ON_MALLOC);
                std::copy(other.getPtr(), other.getEnd(), _items);

            }

            ~MallocArray() {
                std::free(_items);
            }

            size_t getLen() const { return _len; }

            size_t getCap() const { return _cap; }

            bool isFull() const { return _cap == _len; }

            T *getPtr() const { return _items; }

            T *getEnd() const { return _items + _len; }

            T &operator[](size_t index) const {
                if (index > _len) throw MallocArrayException(MallocArrayException::OUT_OF_BOUNDS);
                else return _items[index];
            }

            void push(const T &item) {
                if (isFull()) expand();
                _items[_len++] = item;
            }

            void shrink(size_t smaller) {
                if (smaller >= _len) throw MallocArrayException(MallocArrayException::SHRINK_LARGER);
                _len = smaller;
            }

        private:
            void expand() {
#ifdef PSM_ARRAY_EXP_GROW
                _cap *= _cap;
#else
                _cap *= 2;
#endif // PSM_ARRAY_EXP_GROW
                _items = CPP_REALLOC(_items, T, _cap);
                if (_items == nullptr) throw MallocArrayException(MallocArrayException::NULL_ON_REALLOC);
            }

        private:
            size_t _len;
            size_t _cap;
            T *_items;
        };

/**
 * @brief Allows custom error messages to be formatted from
 */
        struct PackStringException : public std::exception {

            const char *what() const throw() {
                return message;
            }

            char message[256];
        };

/**
 * @class PackString
 * @brief A special string class that allows packing objects into binary formats.
 */
        class PackString {
        public:
            PackString() {}

            ~PackString() {}

            const MallocArray<size_t> &getIndexes() const { return _indexes; }

            size_t dataSize() const { return _data.getLen(); }

            size_t indexesSize() const { return _indexes.getLen(); }

            /* @brief Function that retrieves the start of the binary data pointer.
             * @note Returns pointer to be interfacable with std copy and handlers.
             */
            const unsigned char *bytes() const { return _data.getPtr(); }

            const unsigned char *bytesEnd() const { return _data.getEnd(); }

            /**
             * @brief Writes the bytes of a native C++ object into the PackString
             * @detail According to the result of sizeof on the object, will write each
             *         byte in the objects memory into the PackString.
             * @note When using this with class {} or struct {} defined objects, it will include
             *       all data of the object, even OS padding.
             */
            template<class T>
            void writeObject(const T &item) {
                size_t itemSize = sizeof(item);
                const unsigned char *itemBytes = reinterpret_cast<const unsigned char *>(&item);
                addNextIndex();
                for (size_t i = 0; i < itemSize; i++) _data.push(itemBytes[i]);
            }

            /**
             * @brief An interface writing function meant to be overloaded with type
             *        specific function handlers.
             * @detail This template calls the implemented append_bytes function for the type
             *         passed in as the item. This allows append_bytes functions to be implemented
             *         in different files/translation units.
             */
            template<class T>
            void write(const T &item) {
                addNextIndex();
                append_bytes(*this, item);
            }

            /**
             * @brief Operator function that wraps the write() method for
             *        chainable calls.
             */
            template<class T>
            PackString &operator<<(const T &item) {
                write(item);
                return *this;
            }

            /**
             * @brief This method is primarily intended to be used with implementations of
             *        append_bytes() as friend functions.
             */
            void pushByte(unsigned char byte) { _data.push(byte); }

            /** @brief Allows access to the pakced binary items according to
             *         existing indexes.
             */
            const unsigned char *operator[](size_t index) const;

        private:
            void addNextIndex() { _indexes.push(_data.getLen()); }

        private:
            MallocArray<size_t> _indexes;
            MallocArray<unsigned char> _data;
        };
    }
}

#endif // PACK_STRING_LIB_H