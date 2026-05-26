/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CYLON_DATA_HPP
#define CYLON_DATA_HPP

#include <any>
#include <utility>
#include <vector>
#include <ostream>
#include <functional>
#include <memory>

namespace FMI::Comm {
    //! Small data wrapper around a generic type T with some helper utilities
    template<typename T>
    class Data {
    public:
        Data() = default;
        Data(T &value): val(value) {}
        //Data(T value) : val(value) {}

        std::size_t size_in_bytes() {
            if (std::is_fundamental<T>::value) {
                return sizeof(T);
            } else {
                throw std::runtime_error("Cannot get size in bytes of non-fundamental type");
            }
        }

        //! Gets pointer to the start of the buffer.
        char* data() {
            return reinterpret_cast<char*>(&val);
        }

        //! Returns the value
        T get() const {
            return val;
        }

        friend std::ostream& operator<<( std::ostream& o, const Data& t ) {
            o << t.get();
            return o;
        }

        friend bool operator==(const Data& lhs, const Data& rhs) {
            return lhs.get() == rhs.get();
        }

    private:
        T val;

    };

    //! Wrapper around an arbitrary STL vector
    template<typename A>
    class Data<std::vector<A>> {
    public:
        Data() = default;
        Data(std::size_t n) : val(n) {}
        Data(std::vector<A> value) : val(value) {}

        std::size_t size_in_bytes() {
            return sizeof(A) * val.size();
        }

        //! Pointer to the start of the vector
        char* data() {
            return reinterpret_cast<char*>(val.data());
        }

        //! Returns the vector
        std::vector<A> get() const {
            return val;
        }

    private:
        std::vector<A> val;
    };

    //! Instantiate data with a pointer to memory and an arbitrary size. Should only be used in exceptional cases, the native types should be used otherwise.


    static inline std::function<void(void*)> noop_deleter = [](void*) {};

    template<>
    class Data<void*> {
    public:
        // Default constructor
        Data() = default;

        // Constructor using shared_ptr for ownership management
        Data(std::shared_ptr<void> buf, std::size_t len)
                : buf(std::move(buf)), len(len) {}


        Data(void* buf, std::size_t len, std::function<void(void*)> deleter)
                : buf(std::shared_ptr<void>(buf, std::move(deleter))), len(len) {}


        // Provides access as char*
        char* data() {
            return reinterpret_cast<char*>(buf.get());
        }

        std::size_t size_in_bytes() const {
            return len;

        }

        std::shared_ptr<void> getShared() {
            return buf;
        }

        // Provides raw void* access
        void* get() {
            return buf.get();
        }

    private:
        std::shared_ptr<void> buf;  // Shared pointer for buffer management
        std::size_t len;            // Size of the buffer

    };


}

#endif //CYLON_DATA_HPP
