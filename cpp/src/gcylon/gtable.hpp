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

#ifndef GCYLON_GTABLE_H
#define GCYLON_GTABLE_H

#include <cudf/table/table.hpp>
#include <cudf/io/types.hpp>

#include <cylon/status.hpp>
#include <cylon/ctx/cylon_context.hpp>

namespace gcylon {

class GTable {
public:
    /**
     * constructor with cudf table and the context
     */
    GTable(std::shared_ptr <cylon::CylonContext> &ctx, std::unique_ptr <cudf::table> &tab);

    /**
     * constructor with cudf table, metadata and the context
     */
    GTable(std::shared_ptr <cylon::CylonContext> &ctx,
           std::unique_ptr <cudf::table> &tab,
           cudf::io::table_metadata &metadata);

    /**
     * Create a table from a cudf table,
     * @param table
     * @return
     */
    static cylon::Status FromCudfTable(std::shared_ptr <cylon::CylonContext> &ctx,
                                       std::unique_ptr <cudf::table> &table,
                                       std::shared_ptr <GTable> &tableOut);

    /**
     * Create a table from a cudf table_with_metadata,
     * @param table
     * @return
     */
    static cylon::Status FromCudfTable(std::shared_ptr<cylon::CylonContext> &ctx,
                                        cudf::io::table_with_metadata &table,
                                        std::shared_ptr<GTable> &tableOut);
    /**
     * destructor
     */
    virtual ~GTable();

    /**
     * Returns the cylon Context
     * @return
     */
    std::shared_ptr <cylon::CylonContext> GetContext();

    /**
     * Returns cudf::table
     * @return
     */
    std::unique_ptr<cudf::table> & GetCudfTable();

    /**
     * Returns cudf table metadata
     * @return
     */
    cudf::io::table_metadata & GetCudfMetadata();

    /**
     * sets cudf table metadata
     * @return
     */
    void SetCudfMetadata(cudf::io::table_metadata & metadata);

    //todo: need to add GetTableWithMetadata

private:
    /**
     * Every table should have an unique id
     */
    std::string id_;
    std::shared_ptr <cylon::CylonContext> ctx_;
    std::unique_ptr <cudf::table> table_;
    cudf::io::table_metadata metadata_;
};

}// end of namespace gcylon

#endif //GCYLON_GTABLE_H
