package org.twisterx.io;

import org.apache.arrow.vector.VectorSchemaRoot;
import org.twisterx.NativeLoader;

import java.io.IOException;
import java.util.UUID;

public class Table {

  private VectorSchemaRoot vectorSchemaRoot;
  private String tableId;

  private Table(String tableId) {
    this.tableId = tableId;
  }

//  public static Table fromCSV(Path path, boolean hasHeaders, Types.MinorType... types) throws IOException {
//    CSVParser parse;
//    if (hasHeaders) {
//      parse = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(new FileReader(path.toFile()));
//    } else {
//      parse = CSVFormat.DEFAULT.parse(new FileReader(path.toFile()));
//    }
//
//    RootAllocator rootAllocator = new RootAllocator();
//
//    // create arrow fields
//    List<FieldVector> vectors = new ArrayList<>();
//    for (int i = 0; i < types.length; i++) {
//      String fieldName = hasHeaders ? parse.getHeaderNames().get(i) : "column-" + i;
//      FieldVector vector = types[i].getNewVector(
//          Fields.createDefaultField(fieldName, types[i]),
//          rootAllocator,
//          null
//      );
//      vectors.add(vector);
//      vector.allocateNew();
//    }
//    for (CSVRecord record : parse) {
//      for (int i = 0; i < record.size(); i++) {
//        ValueSetter.parseAndSet(i, record.get(i), types[i], vectors.get(i));
//      }
//    }
//
//    Table table = new Table();
//    table.vectorSchemaRoot = new VectorSchemaRoot(vectors);
//    return table;
//  }

  public static Table fromCSV(String path) {
    String uuid = UUID.randomUUID().toString();
    Table.nativeLoadCSV(path, uuid);
    return new Table(uuid);
  }

  public int getColumnCount() {
    return 0;
  }

  private String getTableId() {
    return tableId;
  }

  public native void join(Table table, int tab1Index, int tab2Index);

  public static native void nativeLoadCSV(String path, String id);

  public static void main(String[] args) throws IOException {
    NativeLoader.load();
    Table table = Table.fromCSV("/tmp/csv.csv");
    System.out.println();
  }
}
