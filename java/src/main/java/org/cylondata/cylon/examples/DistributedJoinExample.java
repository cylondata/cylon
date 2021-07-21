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
import org.cylondata.cylon.ops.JoinConfig;

public class DistributedJoinExample {
  public static void main(String[] args) {
    String src1Path = args[0];
    String src2Path = args[1];

    CylonContext ctx = CylonContext.init();

    Table left = Table.fromCSV(ctx, src1Path);
    Table right = Table.fromCSV(ctx, src2Path);

    JoinConfig joinConfig = new JoinConfig(0, 0);
    Table joined = left.distributedJoin(right, joinConfig);
    joined.print();
    ctx.finalizeCtx();
  }
}
