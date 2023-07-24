#include "PackString.hpp"

namespace cylon {
    namespace util {

        MallocArrayException::MallocArrayException(MallocArrayException::ErrType code) : code(code) {
            switch (code) {
                case NULL_ON_MALLOC:
                    message = "MallocArrayErr: Got nullptr when calling malloc.";
                    break;
                case NULL_ON_REALLOC:
                    message = "MallocArrayErr: Got nullptr when calling realloc.";
                    break;
                case OUT_OF_BOUNDS:
                    message = "MallocArrayErr: Attempted to access an item out of bounds.";
                    break;
                case SHRINK_LARGER:
                    message = "MallocArrayErr: Cannot use shrink() on size larger than current length.";
                    break;
            }
        }


        const unsigned char *PackString::operator[](size_t index) const {
            try {
                return &(_data[_indexes[index]]);
            } catch (const MallocArrayException &exc) {
                PackStringException pexc;
                if (exc.code == MallocArrayException::OUT_OF_BOUNDS) {
                    std::sprintf(pexc.message, "PackString Exception: Index %lu does not exist.\n", index);
                } else {
                    std::sprintf(pexc.message, "Pack String Exception: %s\n", exc.message);
                }
                throw pexc;
            }
        }
    }
}