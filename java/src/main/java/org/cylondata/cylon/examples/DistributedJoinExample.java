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
