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

package org.cylondata.cylon.ops;

import org.cylondata.cylon.Table;

import java.util.List;

/**
 * This class can be used to configure the join operation.
 */
public class JoinConfig {

  private int leftIndex;
  private int rightIndex;
  private List<Integer> leftFilterColumns;
  private List<Integer> rightFilterColumns;
  private Type joinType = Type.INNER;
  private Algorithm joinAlgorithm = Algorithm.SORT;

  /**
   * Types of Joins, analogous to SQL joins
   */
  public enum Type {
    LEFT, RIGHT, INNER, FULL_OUTER
  }

  /**
   * Types of join algorithms
   */
  public enum Algorithm {
    SORT, HASH
  }

  /**
   * Creates and instance of {@link JoinConfig}
   *
   * @param leftIndex  join column index of this left table
   * @param rightIndex join column index of the right table
   */
  public JoinConfig(int leftIndex, int rightIndex) {
    this.leftIndex = leftIndex;
    this.rightIndex = rightIndex;
  }

  /**
   * This method can be used to specify the list of columns to be included in the final join table
   *
   * @param leftFilterColumns  Columns to be included from left table
   * @param rightFilterColumns Columns to be include from right table
   * @return The instance of {@link JoinConfig}
   */
  public JoinConfig filterColumns(List<Integer> leftFilterColumns, List<Integer> rightFilterColumns) {
    this.leftFilterColumns = leftFilterColumns;
    this.rightFilterColumns = rightFilterColumns;
    return this;
  }

  /**
   * Algorithm to be used when joining two {@link Table}s
   *
   * @param algorithm the {@link Algorithm}
   * @return The instance of {@link JoinConfig}
   */
  public JoinConfig useAlgorithm(Algorithm algorithm) {
    this.joinAlgorithm = algorithm;
    return this;
  }

  /**
   * Type of join to perform.
   *
   * @param joinType the {@link Type}
   * @return The instance of {@link JoinConfig}
   */
  public JoinConfig joinType(Type joinType) {
    this.joinType = joinType;
    return this;
  }

  public int getLeftIndex() {
    return leftIndex;
  }

  public int getRightIndex() {
    return rightIndex;
  }

  public Algorithm getJoinAlgorithm() {
    return joinAlgorithm;
  }

  public Type getJoinType() {
    return joinType;
  }
}
