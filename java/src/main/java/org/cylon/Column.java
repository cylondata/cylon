package org.cylon;

/**
 * This class models a column of a {@link Table} instance
 *
 * @param <O>
 */
@SuppressWarnings("unused")
public class Column<O> extends DataRepresentation {

  private int columnIndex = -1;

  /**
   * Creates an instance of Columns
   *
   * @param id uuid of the column. Column ID is different from the column index.
   */
  Column(String id) {
    super(id);
  }

  /**
   * Set the index of the column in a {@link Table}
   *
   * @param columnIndex index of the column
   */
  void setColumnIndex(int columnIndex) {
    this.columnIndex = columnIndex;
  }

  /**
   * Get the index of a column in a table
   *
   * @return index of the column i the {@link Column} already associates with a {@link Table}, -1 if not
   */
  public int getColumnIndex() {
    return columnIndex;
  }

  /**
   * Get the number of rows in this column
   *
   * @return number of rows
   */
  public int getRowCount() {
    throw unSupportedException();
  }
}
