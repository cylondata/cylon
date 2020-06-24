package org.twisterx.examples;

import org.twisterx.Table;
import org.twisterx.TwisterXContext;
import org.twisterx.ops.JoinConfig;

import java.io.IOException;

public class DistributedJoinExample {
  public static void main(String[] args) throws IOException {

    TwisterXContext ctx = TwisterXContext.init();

    long t1 = System.currentTimeMillis();

    Table left = Table.fromCSV(ctx, "/tmp/csv.csv");
    Table right = Table.fromCSV(ctx, "/tmp/csv.csv");

    System.out.println("Data loading time : " + (System.currentTimeMillis() - t1));

    long t2 = System.currentTimeMillis();
    Table joined = left.distributedJoin(right, new JoinConfig(0, 0));
    ctx.barrier();

    System.out.println("Join algorithm time : " + (System.currentTimeMillis() - t2));

    System.out.println("Joined table : " + (joined.getRowCount() + "," + joined.getColumnCount()));
    System.out.println("Total time : " + (System.currentTimeMillis() - t1));

    ctx.finalizeCtx();
  }
}
