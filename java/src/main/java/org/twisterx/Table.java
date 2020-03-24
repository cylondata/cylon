package org.twisterx;

import org.apache.arrow.vector.types.Types;
import org.twisterx.ops.Filter;
import org.twisterx.ops.JoinConfig;
import org.twisterx.ops.Mapper;

import java.util.List;
import java.util.UUID;

/**
 * <p>Table is the basic data manipulation endpoint of TwisterX API. This class doesn't hold any data, instead
 * it acts as the mediator between the user's application and the native TwisterX layer. Data transformation, communication and persistence is handled entirely by the native layer.
 * </p>
 * <p>
 * Tables are immutable and transformations will create another table instance keeping the original table intact.
 * </p>
 * <p>
 * A unique ID will be assigned to each table at the time of creation. This can be considered as the identifier of a set of data.
 * When transferring data to a different machine, or when persisting data to a disk, this ID will be always associated with the underlying set of data. Hence different APIs(Java, Python, REST based RPC) will be able to refer a set of data irrespective of where it actually got created.
 * </p>
 */
@SuppressWarnings({"unused", "rawtypes"})
public class Table extends DataRepresentation {

  /**
   * Creates a new instance of a {@link Table}
   *
   * @param tableId ID of the table
   */
  private Table(String tableId) {
    super(tableId);
  }

  //----------------- METHODS TO GENERATE TABLE ---------------------//

  /**
   * This method will load a table by reading the data from a CSV file.
   *
   * @param path path to the CSV file
   * @return A {@link Table} instance that holds the data from the CSV file
   */
  public static Table fromCSV(String path) {
    String uuid = UUID.randomUUID().toString();
    Table.nativeLoadCSV(path, uuid);
    return new Table(uuid);
  }

  /**
   * Create a {@link Table} by combining a list of columns
   *
   * @param columns {@link List} of columns
   * @return A new table instance which holds the list of columns specified
   */
  public static Table fromColumns(List<Column> columns) {
    throw unSupportedException();
  }

  /**
   * <p>This method will load a table by reading the data from a CSV file. The behaviour will be similar to
   * {@link Table#fromCSV(String)}, but additionally data types can be specified for each column.</p>
   *
   * @param path      path to the CSV file
   * @param dataTypes List of data types, i<sup>th</sup> index of the {@link List} should specify the data types of the i<sup>th</sup> column.
   * @return A {@link Table} instance that holds the data from the CSV file
   */
  public static Table fromCSV(String path, List<Types.MinorType> dataTypes) {
    throw unSupportedException();
  }

  //----------------- END OF METHODS TO GENERATE TABLE ---------------------//


  //----------------- METHODS TO READ TABLE PROPERTIES ---------------------//

  /**
   * Get the number of columns of the table
   *
   * @return No of columns
   */
  public int getColumnCount() {
    return Table.nativeColumnCount(this.getId());
  }

  /**
   * Get the number of rows of the table
   *
   * @return No of rows
   */
  public int getRowCount() {
    return Table.nativeRowCount(this.getId());
  }

  //----------------- END OF METHODS TO READ TABLE PROPERTIES ---------------------//


  //----------------- METHODS FOR TRANSFORMATIONS ---------------------//

  /**
   * Join two tables based on the value of the columns
   *
   * @param rightTable Table to be joined with this table
   * @param joinConfig Configurations for the join operation
   * @return Joined Table
   */
  public Table join(Table rightTable, JoinConfig joinConfig) {
    String uuid = UUID.randomUUID().toString();
    Table.nativeJoin(this.getId(), rightTable.getId(), joinConfig.getLeftIndex(), joinConfig.getRightIndex(), uuid);
    return new Table(uuid);
  }

  /**
   * Maps the values of a column to another value
   *
   * @param colIndex Column index to be transformed(mapped)
   * @param mapper   Mapping logic
   * @param <I>      Input data type
   * @param <O>      Output data type
   * @return An instance of {@link Column} which represents the mapped values
   */
  public <I, O> Column<O> mapColumn(int colIndex, Mapper<I, O> mapper) {
    throw unSupportedException();
  }

  /**
   * Partition a table based on the hash value of the specified columns
   *
   * @param hashColumns    Indices of the columns to be hashed
   * @param noOfPartitions No of partitions to generate
   * @return List of partitioned tables
   */
  public List<Table> hashPartition(List<Integer> hashColumns, int noOfPartitions) {
    throw unSupportedException();
  }

  /**
   * Partition a table into n partitions of similar size
   *
   * @param noOfPartitions no of partitions to generate
   * @return List of partitioned tables
   */
  public List<Table> roundRobinPartition(int noOfPartitions) {
    throw unSupportedException();
  }

  /**
   * Merge a set of similar tables into a single table.
   *
   * @param tables List of tables to be merged
   * @return merged {@link Table}
   */
  public static Table merge(Table... tables) {
    throw unSupportedException();
  }

  /**
   * Sort the rows of a table based on the value of a column
   *
   * @param columnIndex index of the column to be usd for sorting
   * @return Sorted {@link Table} instance
   */
  public Table sort(int columnIndex) {
    throw unSupportedException();
  }

  /**
   * Filter out rows of a table based on a single column
   *
   * @param columnIndex column to be used for filtering
   * @param filterLogic filtering logic
   * @param <I>         data type of the column
   * @return Table without the filtered out rows
   */
  public <I> Table filter(int columnIndex, Filter<I> filterLogic) {
    throw unSupportedException();
  }

  //----------------- END OF METHODS FOR TRANSFORMATIONS ---------------------//

  /**
   * Clear the table and free memory associated with this table
   */
  public void clear() {
    throw unSupportedException();
  }

  //----------------- NATIVE METHODS ---------------------//

  /**
   * @param left        id of the left table
   * @param right       id of the right table
   * @param tab1Index   left join column index
   * @param tab2Index   right join column index
   * @param destination destination table id
   */
  private static native void nativeJoin(String left, String right, int tab1Index, int tab2Index,
                                        String destination);

  private static native int nativeColumnCount(String tableId);

  private static native int nativeRowCount(String tableId);

  private static native void nativeLoadCSV(String path, String id);

  //----------------- END OF METHODS ---------------------//

//  public static void main(String[] args) throws IOException {
//    NativeLoader.load();
//
//    Table left = Table.fromCSV("/tmp/csv.csv");
//    Table right = Table.fromCSV("/tmp/csv.csv");
//    Table joined = left.join(right, new JoinConfig(0, 0));
//
//    System.out.println(joined.getColumnCount());
//    System.out.println(joined.getRowCount());
//  }
}
