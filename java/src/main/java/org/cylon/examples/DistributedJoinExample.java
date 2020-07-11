package org.cylon.examples;

import org.cylon.CylonContext;
import org.cylon.Table;
import org.cylon.ops.JoinConfig;

public class DistributedJoinExample {

  public static void main(String[] args) {

    String src1Path = args[0];
    String src2Path = args[1];

    CylonContext ctx = CylonContext.init();

    Table left = Table.fromCSV(ctx, src1Path);
    Table right = Table.fromCSV(ctx, src2Path);

    Table joined = left.distributedJoin(right, new JoinConfig(0, 0)
        .joinType(JoinConfig.Type.INNER).useAlgorithm(JoinConfig.Algorithm.SORT));
    ctx.barrier();
    System.out.println(String.format("Done Join : Table 1 had %d rows, Table 2 had %d rows. " +
        "Final Join Table has %d rows.", left.getRowCount(), right.getRowCount(), joined.getRowCount()));
    ctx.finalizeCtx();
  }
}
