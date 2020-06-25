package org.twisterx.examples;

import org.twisterx.Table;
import org.twisterx.TwisterXContext;
import org.twisterx.ops.JoinConfig;

import java.io.IOException;

/**
 * mpirun -np 2 java -cp target/twisterx-0.1-SNAPSHOT-jar-with-dependencies.jar org.twisterx.examples.DistributedJoinExample /tmp/csv.csv /tmp/csv.csv 0 0 RIGHT HASH
 */
public class DistributedJoinExample {

  public static void main(String[] args) throws IOException {

    String path1 = args[0];
    String path2 = args[1];
    int table1Column = Integer.parseInt(args[2]);
    int table2Column = Integer.parseInt(args[3]);

    JoinConfig.Type type = JoinConfig.Type.valueOf(args[4]);
    JoinConfig.Algorithm algorithm = JoinConfig.Algorithm.valueOf(args[5]);

    TwisterXContext ctx = TwisterXContext.init();

    long t1 = System.currentTimeMillis();

    Table left = Table.fromCSV(ctx, path1);
    Table right = Table.fromCSV(ctx, path2);

    System.out.println("Data loading time : " + (System.currentTimeMillis() - t1));

    long t2 = System.currentTimeMillis();
    Table joined = left.distributedJoin(right, new JoinConfig(table1Column, table2Column)
        .joinType(type).useAlgorithm(algorithm));
    ctx.barrier();

    System.out.println("Join algorithm time : " + (System.currentTimeMillis() - t2));

    System.out.println("Joined table : " + (joined.getRowCount() + "," + joined.getColumnCount()));
    System.out.println("Total time : " + (System.currentTimeMillis() - t1));

    joined.clear();

    ctx.finalizeCtx();
  }
}
