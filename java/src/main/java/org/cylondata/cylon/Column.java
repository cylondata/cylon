 /*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.cylondata.cylon;

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
