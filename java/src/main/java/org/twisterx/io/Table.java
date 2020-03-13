package org.twisterx.io;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.Types;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.twisterx.io.arrow.Fields;
import org.twisterx.io.arrow.ValueSetter;

import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Table {

  private VectorSchemaRoot vectorSchemaRoot;

  private Table() {

  }

  public static Table fromCSV(Path path, boolean hasHeaders, Types.MinorType... types) throws IOException {
    CSVParser parse;
    if (hasHeaders) {
      parse = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(new FileReader(path.toFile()));
    } else {
      parse = CSVFormat.DEFAULT.parse(new FileReader(path.toFile()));
    }

    RootAllocator rootAllocator = new RootAllocator();

    // create arrow fields
    List<FieldVector> vectors = new ArrayList<>();
    for (int i = 0; i < types.length; i++) {
      String fieldName = hasHeaders ? parse.getHeaderNames().get(i) : "column-" + i;
      FieldVector vector = types[i].getNewVector(
              Fields.createDefaultField(fieldName, types[i]),
              rootAllocator,
              null
      );
      vectors.add(vector);
      vector.allocateNew();
    }
    for (CSVRecord record : parse) {
      for (int i = 0; i < record.size(); i++) {
        ValueSetter.parseAndSet(i, record.get(i), types[i], vectors.get(i));
      }
    }

    Table table = new Table();
    table.vectorSchemaRoot = new VectorSchemaRoot(vectors);
    return table;
  }

  public static void main(String[] args) throws IOException {
    Table table = Table.fromCSV(Paths.get("/tmp/csv.csv"), true, Types.MinorType.INT, Types.MinorType.INT, Types.MinorType.INT);
    System.out.println();
  }
}
