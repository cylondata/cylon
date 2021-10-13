// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "common/test_header.hpp"
#include "test_arrow_utils.hpp"
#include "test_macros.hpp"

#include "cylon/util/flatten_array.hpp"

namespace cylon {
namespace test {

TEST_CASE("Test single array non null", "[flatten array]") {
  SECTION("int") {
    auto a1 = ArrayFromJSON(arrow::int32(), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
    std::shared_ptr<FlattenedArray> flattened;
    CHECK_CYLON_STATUS(FlattenArrays(ctx.get(), {a1}, &flattened));
    CHECK_ARROW_BUFFER_EQUAL(a1->data()->buffers[1], flattened->data->data()->buffers[2]);
  }

  SECTION("str") {
    auto a1 = ArrayFromJSON(arrow::utf8(),
                            R"(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])");
    std::shared_ptr<FlattenedArray> flattened;
    CHECK_CYLON_STATUS(FlattenArrays(ctx.get(), {a1}, &flattened));
    CHECK_ARROW_BUFFER_EQUAL(a1->data()->buffers[1], flattened->data->data()->buffers[1]);
    CHECK_ARROW_BUFFER_EQUAL(a1->data()->buffers[2], flattened->data->data()->buffers[2]);
  }
}

TEST_CASE("Test numeric", "[flatten array]") {
  auto a1 = ArrayFromJSON(arrow::int32(), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
  auto a2 = ArrayFromJSON(arrow::int8(), "[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]");
  auto a3 = ArrayFromJSON(arrow::int64(),
                          "[1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]");

  std::shared_ptr<FlattenedArray> flattened;
  CHECK_CYLON_STATUS(FlattenArrays(ctx.get(), {a1, a2, a3}, &flattened));
  /*
   # byte string was generated from this python code
    a1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    a2 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.int8)
    a3 = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000], dtype=np.int64)

    for x in zip(a1, a2, a3):
      print("{},".format(', '.join(['0x{:02x}'.format(c) for c in b''.join([a.tobytes() for a in x])])))
   */

  std::vector<uint8_t>
      expected{0x00, 0x00, 0x00, 0x00, 0x0a, 0xe8, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
               0x01, 0x00, 0x00, 0x00, 0x0b, 0x4c, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
               0x02, 0x00, 0x00, 0x00, 0x0c, 0xb0, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
               0x03, 0x00, 0x00, 0x00, 0x0d, 0x14, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
               0x04, 0x00, 0x00, 0x00, 0x0e, 0x78, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
               0x05, 0x00, 0x00, 0x00, 0x0f, 0xdc, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
               0x06, 0x00, 0x00, 0x00, 0x10, 0x40, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
               0x07, 0x00, 0x00, 0x00, 0x11, 0xa4, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
               0x08, 0x00, 0x00, 0x00, 0x12, 0x08, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
               0x09, 0x00, 0x00, 0x00, 0x13, 0x6c, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
               0x0a, 0x00, 0x00, 0x00, 0x14, 0xd0, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

  auto expected_buf =
      std::make_shared<arrow::Buffer>(expected.data(), static_cast<int64_t>(expected.size()));
  CHECK_ARROW_BUFFER_EQUAL(expected_buf, flattened->data->data()->buffers[2]);
}

TEST_CASE("Test string", "[flatten array]") {
  auto a1 = ArrayFromJSON(arrow::utf8(),
                          R"(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])");
  auto a2 = ArrayFromJSON(arrow::utf8(),
                          R"(["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"])");
  auto a3 = ArrayFromJSON(
      arrow::utf8(),
      R"(["1000", "1100", "1200", "1300", "1400", "1500", "1600", "1700", "1800", "1900", "2000"])"
  );

  std::shared_ptr<FlattenedArray> flattened;
  CHECK_CYLON_STATUS(FlattenArrays(ctx.get(), {a1, a2, a3}, &flattened));

  /*
    # byte string was generated from this python code
    a1 = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    a2 = np.array(["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"])
    a3 = np.array(["1000", "1100", "1200", "1300", "1400", "1500", "1600", "1700", "1800", "1900", "2000"])
    for x in zip(a1, a2, a3):
      print("{},".format("".join(x)))
  */

  auto expected = ArrayFromJSON(arrow::binary(), R"(["0101000",
                                                                "1111100",
                                                                "2121200",
                                                                "3131300",
                                                                "4141400",
                                                                "5151500",
                                                                "6161600",
                                                                "7171700",
                                                                "8181800",
                                                                "9191900",
                                                                "10202000"])");
  CHECK_ARROW_EQUAL(expected, flattened->data);
}

TEST_CASE("Test numeric & string", "[flatten array]") {
  auto a1 = ArrayFromJSON(arrow::utf8(),
                          R"(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])");
  auto a2 = ArrayFromJSON(arrow::int32(), "[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]");
  auto a3 = ArrayFromJSON(arrow::utf8(),
                          R"(["1000", "1100", "1200", "1300", "1400", "1500", "1600", "1700", "1800", "1900", "2000"])");

  std::shared_ptr<FlattenedArray> flattened;
  CHECK_CYLON_STATUS(FlattenArrays(ctx.get(), {a1, a2, a3}, &flattened));

  /*
    # byte string was generated from this python code
    a1 = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    a2 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.int32)
    a3 = np.array(["1000", "1100", "1200", "1300", "1400", "1500", "1600", "1700", "1800", "1900", "2000"])
    for x in zip(a1, a2, a3):
      print("{}, {}, {},".format(", ".join(['0x{:02x}'.format(ord(c)) for c in x[0]]),
                                 ", ".join(['0x{:02x}'.format(c) for c in x[1].tobytes()]),
                                 ", ".join(['0x{:02x}'.format(ord(c)) for c in x[2]])))
   */

  std::vector<uint8_t> expected{0x30, 0x0a, 0x00, 0x00, 0x00, 0x31, 0x30, 0x30, 0x30,
                                0x31, 0x0b, 0x00, 0x00, 0x00, 0x31, 0x31, 0x30, 0x30,
                                0x32, 0x0c, 0x00, 0x00, 0x00, 0x31, 0x32, 0x30, 0x30,
                                0x33, 0x0d, 0x00, 0x00, 0x00, 0x31, 0x33, 0x30, 0x30,
                                0x34, 0x0e, 0x00, 0x00, 0x00, 0x31, 0x34, 0x30, 0x30,
                                0x35, 0x0f, 0x00, 0x00, 0x00, 0x31, 0x35, 0x30, 0x30,
                                0x36, 0x10, 0x00, 0x00, 0x00, 0x31, 0x36, 0x30, 0x30,
                                0x37, 0x11, 0x00, 0x00, 0x00, 0x31, 0x37, 0x30, 0x30,
                                0x38, 0x12, 0x00, 0x00, 0x00, 0x31, 0x38, 0x30, 0x30,
                                0x39, 0x13, 0x00, 0x00, 0x00, 0x31, 0x39, 0x30, 0x30,
                                0x31, 0x30, 0x14, 0x00, 0x00, 0x00, 0x32, 0x30, 0x30, 0x30};

  auto expected_buf =
      std::make_shared<arrow::Buffer>(expected.data(), static_cast<int64_t>(expected.size()));
  CHECK_ARROW_BUFFER_EQUAL(expected_buf, flattened->data->data()->buffers[2]);
}

TEST_CASE("Test numeric w/ nulls", "[flatten array]") {
  auto a1 = ArrayFromJSON(arrow::int32(), "[0, null, 2, 3, null, 5, 6, 7, 8, null, 10]");
  auto a2 = ArrayFromJSON(arrow::int8(), "[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]");
  auto a3 = ArrayFromJSON(arrow::int64(),
                          "[1000, null, 1200, 1300, 1400, null, 1600, 1700, 1800, null, 2000]");

  std::shared_ptr<FlattenedArray> flattened;
  CHECK_CYLON_STATUS(FlattenArrays(ctx.get(), {a1, a2, a3}, &flattened));

  /*
   # byte string was generated from this python code
  a1 = [0, None, 2, 3, None, 5, 6, 7, 8, None, 10]
  a2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
  a3 = [1000, None, 1200, 1300, 1400, None, 1600, 1700, 1800, None, 2000]
  byte_witdths = [4, 1, 8]
  for x in zip(a1, a2, a3):
    s = ""
    nulls = [0]
    for i in range(len(x)):
      if x[i] is None:
        nulls[0]+=1
        nulls.append(i)
    s += ", ".join(['0x{:02x}'.format(c) for c in nulls])
    for i in range(len(x)):
      a = x[i] if x[i] is not None else 0
      s += ", {}".format(", ".join(['0x{:02x}'.format(c) for c in a.to_bytes(byte_witdths[i], 'little')]))
    print("{},".format(s))
  */

  std::vector<uint8_t> expected{
      0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0xe8, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00,
      0x00, 0x02, 0x00, 0x00, 0x00, 0x0c, 0xb0, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x03, 0x00, 0x00, 0x00, 0x0d, 0x14, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x78, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x02, 0x05, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x06, 0x00, 0x00, 0x00, 0x10, 0x40, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x07, 0x00, 0x00, 0x00, 0x11, 0xa4, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x08, 0x00, 0x00, 0x00, 0x12, 0x08, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00,
      0x00, 0x0a, 0x00, 0x00, 0x00, 0x14, 0xd0, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

  auto expected_buf =
      std::make_shared<arrow::Buffer>(expected.data(), static_cast<int64_t>(expected.size()));
  CHECK_ARROW_BUFFER_EQUAL(expected_buf, flattened->data->data()->buffers[2]);
}

TEST_CASE("Test string w/ nulls", "[flatten array]") {
  auto a1 = ArrayFromJSON(arrow::utf8(),
                          R"(["a", null, "b", "c", null, "d", "e", "f", "g", null, "h"])");
  auto a2 = ArrayFromJSON(arrow::utf8(),
                          R"(["qq", "ww", "ee", "rr", "tt", "yy", "uu", "ii", "oo", "pp", "aa"])");
  auto a3 = ArrayFromJSON(
      arrow::utf8(),
      R"(["aaa", null, "sss", "ddd", "fff", null, "ggg", "hhh", "jjj", null, "kkk"])");

  std::shared_ptr<FlattenedArray> flattened;
  CHECK_CYLON_STATUS(FlattenArrays(ctx.get(), {a1, a2, a3}, &flattened));

  /*
    # byte string was generated from this python code
    a1 = ["a", None, "b", "c", None, "d", "e", "f", "g", None, "h"]
    a2 = ["qq", "ww", "ee", "rr", "tt", "yy", "uu", "ii", "oo", "pp", "aa"]
    a3 = ["aaa", None, "sss", "ddd", "fff", None, "ggg", "hhh", "jjj", None, "kkk"]
    for x in zip(a1, a2, a3):
      s = ""
      nulls = [0]
      for i in range(len(x)):
        if x[i] is None:
          nulls[0]+=1
          nulls.append(i)
      s += ", ".join(['0x{:02x}'.format(c) for c in nulls])
      for a in x:
        if a is not None:
          s += ", {}".format(", ".join(['0x{:02x}'.format(ord(c)) for c in a]))
      print("{},".format(s))
  */

  std::vector<uint8_t> expected{0x00, 0x61, 0x71, 0x71, 0x61, 0x61, 0x61,
                                0x02, 0x00, 0x02, 0x77, 0x77,
                                0x00, 0x62, 0x65, 0x65, 0x73, 0x73, 0x73,
                                0x00, 0x63, 0x72, 0x72, 0x64, 0x64, 0x64,
                                0x01, 0x00, 0x74, 0x74, 0x66, 0x66, 0x66,
                                0x01, 0x02, 0x64, 0x79, 0x79,
                                0x00, 0x65, 0x75, 0x75, 0x67, 0x67, 0x67,
                                0x00, 0x66, 0x69, 0x69, 0x68, 0x68, 0x68,
                                0x00, 0x67, 0x6f, 0x6f, 0x6a, 0x6a, 0x6a,
                                0x02, 0x00, 0x02, 0x70, 0x70,
                                0x00, 0x68, 0x61, 0x61, 0x6b, 0x6b, 0x6b};
  auto expected_buf =
      std::make_shared<arrow::Buffer>(expected.data(), static_cast<int64_t>(expected.size()));
  CHECK_ARROW_BUFFER_EQUAL(expected_buf, flattened->data->data()->buffers[2]);
}

TEST_CASE("Test numeric & string w/ nulls", "[flatten array]") {
  auto a1 = ArrayFromJSON(arrow::int32(), "[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]");
  auto a2 = ArrayFromJSON(
      arrow::utf8(),
      R"(["aaa", null, "sss", "ddd", null, "fff", "ggg", "hhh", "jjj", null, "kkk"])");
  auto a3 = ArrayFromJSON(
      arrow::utf8(), R"(["qq", "ww", "ee", "rr", "tt", "yy", "uu", "ii", "oo", "pp", "aa"])");
  auto a4 = ArrayFromJSON(arrow::int64(),
                          "[1000, null, 1200, 1300, 1400, null, 1600, 1700, 1800, null, 2000]");

  std::shared_ptr<FlattenedArray> flattened;
  CHECK_CYLON_STATUS(FlattenArrays(ctx.get(), {a1, a2, a3, a4}, &flattened));

  /*
    # byte string was generated from this python code
    a1 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    a2 = ["aaa", None, "sss", "ddd", None, "fff", "ggg", "hhh", "jjj", None, "kkk"]
    a3 = ["qq", "ww", "ee", "rr", "tt", "yy", "uu", "ii", "oo", "pp", "aa"]
    a4 = [1000, None, 1200, 1300, 1400, None, 1600, 1700, 1800, None, 2000]
    byte_widths = [4, None, None, 8]

    for x in zip(a1, a2, a3, a4):
      s = ""
      nulls = [0]
      for i in range(len(x)):
        if x[i] is None:
          nulls[0]+=1
          nulls.append(i)
      s += ", ".join(['0x{:02x}'.format(c) for c in nulls])
      for i in range(len(x)):
        if byte_widths[i] is not None: # int
          a = x[i] if x[i] is not None else 0
          s += ", {}".format(", ".join(['0x{:02x}'.format(c)
                                for c in a.to_bytes(byte_widths[i], 'little')]))
        elif x[i] is not None: # str
          s += ", {}".format(", ".join(['0x{:02x}'.format(ord(c)) for c in x[i]]))
      print("{},".format(s))
  */

  std::vector<uint8_t> expected{
      0x00, 0x0a, 0x00, 0x00, 0x00, 0x61, 0x61, 0x61, 0x71, 0x71, 0xe8, 0x03, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00,
      0x02, 0x01, 0x03, 0x0b, 0x00, 0x00, 0x00, 0x77, 0x77, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00,
      0x00, 0x0c, 0x00, 0x00, 0x00, 0x73, 0x73, 0x73, 0x65, 0x65, 0xb0, 0x04, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00,
      0x00, 0x0d, 0x00, 0x00, 0x00, 0x64, 0x64, 0x64, 0x72, 0x72, 0x14, 0x05, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00,
      0x01, 0x01, 0x0e, 0x00, 0x00, 0x00, 0x74, 0x74, 0x78, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00,
      0x01, 0x03, 0x0f, 0x00, 0x00, 0x00, 0x66, 0x66, 0x66, 0x79, 0x79, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
      0x00, 0x10, 0x00, 0x00, 0x00, 0x67, 0x67, 0x67, 0x75, 0x75, 0x40, 0x06, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00,
      0x00, 0x11, 0x00, 0x00, 0x00, 0x68, 0x68, 0x68, 0x69, 0x69, 0xa4, 0x06, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00,
      0x00, 0x12, 0x00, 0x00, 0x00, 0x6a, 0x6a, 0x6a, 0x6f, 0x6f, 0x08, 0x07, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00,
      0x02, 0x01, 0x03, 0x13, 0x00, 0x00, 0x00, 0x70, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00,
      0x00, 0x14, 0x00, 0x00, 0x00, 0x6b, 0x6b, 0x6b, 0x61, 0x61, 0xd0, 0x07, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00};

  auto expected_buf =
      std::make_shared<arrow::Buffer>(expected.data(), static_cast<int64_t>(expected.size()));
  CHECK_ARROW_BUFFER_EQUAL(expected_buf, flattened->data->data()->buffers[2]);
}

}
}