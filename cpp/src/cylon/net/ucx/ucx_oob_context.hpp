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

#ifndef CYLON_UCX_OOB_CONTEXT_HPP
#define CYLON_UCX_OOB_CONTEXT_HPP


#include "cylon/status.hpp"

namespace cylon {
    namespace net {
        class UCXOOBContext {
        public:
            virtual Status InitOOB() = 0;

            virtual Status getWorldSizeAndRank(int &world_size, int &rank) = 0;

            virtual Status OOBAllgather(uint8_t *src, uint8_t *dst, size_t srcSize,
                                        size_t dstSize) = 0;

            virtual Status Finalize() = 0;
        };
    }
}

#endif //CYLON_UCX_OOB_CONTEXT_HPP
