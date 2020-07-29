package org.cylondata.cylon;

import org.apache.arrow.vector.types.Types;
import org.cylondata.cylon.arrow.ArrowTable;
import org.cylondata.cylon.exception.CylonRuntimeException;
import org.cylondata.cylon.ops.Filter;
import org.cylondata.cylon.ops.JoinConfig;
import org.cylondata.cylon.ops.Mapper;
import org.cylondata.cylon.ops.Selector;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
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
public class Table extends DataRepresentation implements Clearable {

  private CylonContext ctx;
  private Set<Clearable> clearables;

  /**
   * Creates a new instance of a {@link Table}
   *
   * @param tableId ID of the table
   */
  private Table(String tableId, CylonContext ctx) {
    super(tableId);
    this.ctx = ctx;
    this.clearables = new HashSet<>();
  }

  //----------------- METHODS TO GENERATE TABLE ---------------------//

  public static Table fromArrowTable(CylonContext ctx, ArrowTable arrowTable) {
    if (!arrowTable.isFinished()) {
      throw new CylonRuntimeException("Can't create a Table from an unfinished arrow table");
    }
    Table table = new Table(arrowTable.getUuid(), ctx);
    table.clearables.add(arrowTable);
    arrowTable.markReferred();
    return table;
  }

  /**
   * This method will load a table by reading the data from a CSV file.
   *
   * @param path path to the CSV file
   * @return A {@link Table} instance that holds the data from the CSV file
   */
  public static Table fromCSV(CylonContext ctx, String path) {
    String uuid = UUID.randomUUID().toString();
    Table.nativeLoadCSV(ctx.getCtxId(), path, uuid);
    return new Table(uuid, ctx);
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
   * {@link Table#fromCSV(CylonContext, String)}, but additionally data types can be specified for each column.</p>
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
    Table.nativeJoin(this.ctx.getCtxId(), this.getId(), rightTable.getId(), joinConfig.getLeftIndex(),
        joinConfig.getRightIndex(), joinConfig.getJoinType().name(), joinConfig.getJoinAlgorithm().name(), uuid);
    return new Table(uuid, this.ctx);
  }

  /**
   * Apply the join algorithm across a distributed set of nodes/dataset
   *
   * @param rightTable Table to be joined with this table
   * @param joinConfig Configurations for the join operation
   * @return Joined Table
   */
  public Table distributedJoin(Table rightTable, JoinConfig joinConfig) {
    String uuid = UUID.randomUUID().toString();
    Table.nativeDistributedJoin(this.ctx.getCtxId(), this.getId(), rightTable.getId(), joinConfig.getLeftIndex(),
        joinConfig.getRightIndex(), joinConfig.getJoinType().name(), joinConfig.getJoinAlgorithm().name(), uuid);
    return new Table(uuid, this.ctx);
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
  public static Table merge(CylonContext ctx, Table... tables) {
    String[] tableIds = new String[tables.length];
    for (int i = 0; i < tables.length; i++) {
      tableIds[i] = tables[i].getId();
    }
    String uuid = UUID.randomUUID().toString();
    merge(ctx.getCtxId(), tableIds, uuid);
    return new Table(uuid, ctx);
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

  /**
   * This method can be used to filter out some rows from a table based on a
   * user defined logic
   *
   * @param selector logic to select(filter) rows from the table
   * @return resulting table, after filtering out rows
   */
  public Table select(Selector selector) {
    String destination = UUID.randomUUID().toString();
    Table.select(this.ctx.getCtxId(), this.getId(), selector, destination);
    return new Table(destination, this.ctx);
  }

  //----------------- END OF METHODS FOR TRANSFORMATIONS ---------------------//

  /**
   * Clear the table and free memory associated with this table
   */
  @Override
  public void clear() {
    for (Clearable clearable : this.clearables) {
      clearable.clear();
    }
    Table.clear(this.getId());
  }

  /**
   * Prints the entire table to the console
   */
  public void print() {
    print(this.getId(), 0, this.getRowCount(), 0, this.getColumnCount());
  }

  /**
   * Prints a section of the table to the console
   *
   * @param row1 starting row index
   * @param row2 ending row index
   * @param col1 starting column index
   * @param col2 ending column index
   */
  public void print(int row1, int row2, int col1, int col2) {
    print(this.getId(), row1, row2, col1, col2);
  }

  //----------------- NATIVE METHODS ---------------------//

  /**
   * @param left          id of the left table
   * @param right         id of the right table
   * @param tab1Index     left join column index
   * @param tab2Index     right join column index
   * @param joinType      join type
   * @param joinAlgorithm join algorithm
   * @param destination   destination table id
   */
  private static native void nativeJoin(int ctxId, String left, String right, int tab1Index, int tab2Index,
                                        String joinType, String joinAlgorithm, String destination);

  private static native void nativeDistributedJoin(int ctxId, String left, String right, int tab1Index, int tab2Index,
                                                   String joinType, String joinAlgorithm, String destination);

  private static native int nativeColumnCount(String tableId);

  private static native int nativeRowCount(String tableId);

  private static native void nativeLoadCSV(int ctxId, String path, String id);

  private static native void print(String tableId, int row1, int row2, int col1, int col2);

  private static native void merge(int ctxId, String[] tableIds, String mergedTableId);

  private static native void clear(String id);

  private static native void select(int ctx, String tableId, Selector selector, String destination);

  //----------------- END OF METHODS ---------------------//
}
