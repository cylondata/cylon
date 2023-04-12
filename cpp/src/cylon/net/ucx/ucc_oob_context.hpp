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

#ifndef CYLON_UCC_OOB_CONTEXT_HPP
#define CYLON_UCC_OOB_CONTEXT_HPP

#include <cylon/net/comm_config.hpp>
#include <cylon/net/communicator.hpp>
#include <cylon/net/ucx/ucx_operations.hpp>
#include <cylon/net/ucx/oob_type.hpp>

#include "cylon/util/macros.hpp"

#include <cylon/net/ucx/ucx_ucc_oob_context.hpp>


namespace cylon {
    namespace net {
#ifdef BUILD_CYLON_UCC

        class UCCOOBContext {
        public:
            virtual OOBType Type() = 0;

            virtual std::shared_ptr <UCXOOBContext> makeUCXOOBContext() = 0;

            virtual void InitOOB(int rank) = 0;

            virtual void *getCollInfo() = 0;
        };

#endif
    }
}

#endif //CYLON_UCC_OOB_CONTEXT_HPP
