package org.twisterx.examples;

import org.twisterx.Table;
import org.twisterx.TwisterXContext;
import org.twisterx.ops.JoinConfig;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class DistributedJoinExample {

  public static void main(String[] args) throws IOException {

    String srcPath = args[0];
    String dstPath = args[1];

    int table1Column = 0;
    int table2Column = 0;
    JoinConfig.Type type = JoinConfig.Type.INNER;

    TwisterXContext ctx = TwisterXContext.init();

    Path csv1FileSrc = Paths.get(srcPath, "csv1_" + ctx.getRank() + ".csv");
    Path csv2FileSrc = Paths.get(srcPath, "csv1_" + ctx.getRank() + ".csv");

    File destinationFile = new File(dstPath + "/data");
    if (!destinationFile.mkdirs()) {
      throw new RuntimeException("Failed to create destination directories");
    }

    File csv1File = new File(dstPath + "/data/csv1.csv");
    File csv2File = new File(dstPath + "/data/csv2.csv");

    csv1File.deleteOnExit();
    csv2File.deleteOnExit();

    System.out.println("Copying files to " + destinationFile.getAbsolutePath());
    Files.copy(csv1FileSrc, new FileOutputStream(csv1File));
    Files.copy(csv2FileSrc, new FileOutputStream(csv2File));
    System.out.println("Copied files.");

    Table left = Table.fromCSV(ctx, csv1File.getAbsolutePath());
    Table right = Table.fromCSV(ctx, csv2File.getAbsolutePath());

    for (JoinConfig.Algorithm algorithm : JoinConfig.Algorithm.values()) {
      System.out.println("Starting Join : " + algorithm.name());
      long t1 = System.currentTimeMillis();
      Table joined = left.distributedJoin(right, new JoinConfig(table1Column, table2Column)
          .joinType(type).useAlgorithm(JoinConfig.Algorithm.SORT));
      ctx.barrier();
      System.out.println(String.format("TOKEN %d j_t %d w_t %d lines %d t 0 a %d",
          ctx.getRank(),
          (System.currentTimeMillis() - t1),
          0,
          joined.getRowCount(),
          algorithm.ordinal()
      ));
      joined.clear();
      System.out.println("Done Join : " + algorithm.name());
    }
    ctx.finalizeCtx();
  }
}
