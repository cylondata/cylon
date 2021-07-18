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

public enum DataTypes {

  BIGINT(0),
  BIT(1),
  DATEDAY(2),
  DATEMILLI(3),
  DECIMAL(4),
  DURATION(5),
  EXTENSIONTYPE(38),
  FIXED_SIZE_LIST(6),
  FIXEDSIZEBINARY(7),
  FLOAT4(8),
  FLOAT8(9),
  INT(10),
  INTERVALDAY(11),
  INTERVALYEAR(12),
  LIST(13),
  MAP(14),
  NULL(15),
  SMALLINT(16),
  STRUCT(17),
  TIMEMICRO(18),
  TIMEMILLI(19),
  TIMENANO(20),
  TIMESEC(21),
  TIMESTAMPMICRO(22),
  TIMESTAMPMICROTZ(23),
  TIMESTAMPMILLI(24),
  TIMESTAMPMILLITZ(25),
  TIMESTAMPNANO(26),
  TIMESTAMPNANOTZ(27),
  TIMESTAMPSEC(28),
  TIMESTAMPSECTZ(29),
  TINYINT(30),
  UINT1(31),
  UINT2(32),
  UINT4(33),
  UINT8(34),
  UNION(35),
  VARBINARY(36),
  VARCHAR(37);

  private int idx;

  DataTypes(int idx) {
    this.idx = idx;
  }

  public int getIdx() {
    return idx;
  }
}
