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

#ifndef GCYLON_GTABLE_API_H
#define GCYLON_GTABLE_API_H

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/io/types.hpp>

#include <gcylon/gtable.hpp>
#include <gcylon/all2all/cudf_all_to_all.hpp>
#include <cylon/join/join_config.hpp>

namespace gcylon {

/**
 * Shuffles a cudf::table with table_view
 * this is to be called from cython code and the other Shuffle with GTable
 * @param inputTable
 * @param columns_to_hash
 * @param ctx
 * @param table_out
 * @return
 */
cylon::Status Shuffle(const cudf::table_view & inputTable,
                      const std::vector<int> &columns_to_hash,
                      std::shared_ptr<cylon::CylonContext> ctx,
                      std::unique_ptr<cudf::table> &table_out);

/**
 * Similar to local join, but performs the join in a distributed fashion
 * @param left
 * @param right
 * @param join_config
 * @param output
 * @return <cylon::Status>
 */
cylon::Status DistributedJoin(const cudf::table_view & leftTable,
                              const cudf::table_view & rightTable,
                              const cylon::join::config::JoinConfig &join_config,
                              std::shared_ptr<cylon::CylonContext> ctx,
                              std::unique_ptr<cudf::table> &table_out);


/**
* Shuffles a GTable based on hashes of the given columns
* @param table
* @param hash_col_idx vector of column indicies that needs to be hashed
* @param output
* @return
*/
cylon::Status Shuffle(std::shared_ptr<GTable> &inputTable,
                      const std::vector<int> &columns_to_hash,
                      std::shared_ptr<GTable> &outputTable);

/**
 * Similar to local join, but performs the join in a distributed fashion
 * @param left
 * @param right
 * @param join_config
 * @param output
 * @return <cylon::Status>
 */
cylon::Status DistributedJoin(std::shared_ptr<GTable> &left,
                       std::shared_ptr<GTable> &right,
                       const cylon::join::config::JoinConfig &join_config,
                       std::shared_ptr<GTable> &output);

/**
 * write the csv table to a file
 * @param table
 * @param outputFile
 * @return
 */
cylon::Status WriteToCsv(std::shared_ptr<GTable> &table, std::string outputFile);

}// end of namespace gcylon

#endif //GCYLON_GTABLE_API_H
