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

package org.cylondata.cylon.examples;

import org.cylondata.cylon.CylonContext;
import org.cylondata.cylon.Table;

public class SelectExample {
  public static void main(String[] args) {
    String tablePath = args[0];

    // initialize Cylon context
    CylonContext ctx = CylonContext.init();

    // create a table from csv
    Table srcTable = Table.fromCSV(ctx, tablePath);

    final long somethingOutside = 7;

    // applying select operation
    Table select = srcTable.select((row) -> row.getInt64(0) == somethingOutside);

    // print the table to console
    select.print();

    // finalizing Cylon context
    ctx.finalizeCtx();
  }
}
