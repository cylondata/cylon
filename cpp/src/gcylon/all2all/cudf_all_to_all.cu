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

#include <gcylon/all2all/cudf_all_to_all.cuh>

namespace gcylon {

__global__ void rebaseOffsets(int32_t * arr, int size, int32_t base) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        arr[i] -= base;
    }
}

int ceil(const int& numerator, const int& denominator) {
   return (numerator + denominator - 1) / denominator;
}

//todo: may be optimized better.
//      each thread can rebase a set of offsets instead of one
void callRebaseOffsets(int32_t * arr, int size, int32_t base){
    int threads_per_block = 256;
    int number_of_blocks = ceil(size, threads_per_block);
    rebaseOffsets<<<number_of_blocks, threads_per_block>>>(arr, size, base);
    cudaDeviceSynchronize();
}

}// end of namespace gcylon
