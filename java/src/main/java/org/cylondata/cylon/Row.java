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

public class Row {

  // this field will be read/write by CPP only
  private long memoryAddress;

  public Row() {
  }

  public native byte getInt8(long column);

  public native short getUInt8(long column);

  public native short getInt16(long column);

  public native int getUInt16(long column);

  public native int getInt32(long column);

  public native long getUInt32(long column);

  public native long getInt64(long column);

  public native long getUInt64(long column);

  public native float getHalfFloat(long column);

  public native float getFloat(long column);

  public native double getDouble(long column);

  public native String getString(long column);
}
