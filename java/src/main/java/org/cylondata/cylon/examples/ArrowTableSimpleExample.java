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

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.util.Text;
import org.cylondata.cylon.CylonContext;
import org.cylondata.cylon.NativeLoader;
import org.cylondata.cylon.Table;
import org.cylondata.cylon.arrow.ArrowTable;

public class ArrowTableSimpleExample {

  public static void main(String[] args) {
    NativeLoader.load();

    // build arrow stuff

    RootAllocator rootAllocator = new RootAllocator();
    IntVector intVector = new IntVector("col1", rootAllocator);
    intVector.allocateNew(200);

    Float8Vector float8Vector = new Float8Vector("col2", rootAllocator);
    float8Vector.allocateNew(200);


    VarCharVector stringVector = new VarCharVector("col3", rootAllocator);


    for (int i = 0; i < 200; i++) {
      float8Vector.setSafe(i, i);

      if (i % 10 == 0) {
        intVector.setNull(i);
      } else {
        intVector.setSafe(i, i);
      }

      stringVector.setSafe(i, new Text("hello" + i));
    }

    intVector.setValueCount(200);
    float8Vector.setValueCount(200);
    stringVector.setValueCount(200);

    // convert to Cylon

    ArrowTable arrowTable = new ArrowTable();
    arrowTable.addColumn("col1", intVector);
    arrowTable.addColumn("col2", float8Vector);
    arrowTable.addColumn("col3", stringVector);
    arrowTable.finish();

    CylonContext ctx = CylonContext.init();
    Table table = Table.fromArrowTable(ctx, arrowTable);
    table.print();
  }
}
