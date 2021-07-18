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

@SuppressWarnings("unused")
public class DataRepresentation {

  private String id;

  public DataRepresentation(String id) {
    this.id = id;
  }

  protected static UnsupportedOperationException unSupportedException() {
    return new UnsupportedOperationException("This operation is not supported yet");
  }

  /**
   * Returns the unique ID assigned for this {@link DataRepresentation}.
   * This can be useful when cross API scripting is required.
   *
   * @return ID
   */
  public String getId() {
    return id;
  }
}
