//
// Created by vibhatha on 4/17/20.
//

#ifndef TWISTERX_BUILTINS_H
#define TWISTERX_BUILTINS_H

#include "iostream"

using namespace std;

namespace twisterx{
    namespace util{

        void printArray(void *buf, int size, string dataType, int depth);

        void printInt8Array(int8_t * buf, int size);

        void printInt16Array(int16_t * buf, int size);

        void printInt32Array(int32_t * buf, int size);

        void printInt64Array(int64_t * buf, int size);

        void printFloatArray(float * buf, int size);

        void printDoubleArray(double * buf, int size);
    }
}




#endif //TWISTERX_BUILTINS_H
