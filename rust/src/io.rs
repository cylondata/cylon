// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! I/O operations for reading and writing data
//!
//! Ported from cpp/src/cylon/io/

pub mod csv;

#[cfg(feature = "parquet")]
pub mod arrow_io;
#[cfg(feature = "parquet")]
pub mod parquet_config;

pub use csv::{CsvReadOptions, CsvWriteOptions, read_csv, write_csv};

#[cfg(feature = "parquet")]
pub use arrow_io::{read_parquet, write_parquet};
#[cfg(feature = "parquet")]
pub use parquet_config::ParquetOptions;