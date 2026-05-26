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

//! Cylon: Fast, scalable distributed memory data parallel library
//!
//! Cylon is a distributed computing library that provides high-performance
//! data processing capabilities for structured data. It implements relational
//! operators and uses Apache Arrow as the underlying data format.

pub mod arrow;
#[cfg(feature = "runtime")]
pub mod checkpoint;
pub mod compute;
pub mod ctx;
pub mod data_types;
pub mod error;
pub mod groupby;
pub mod indexing;
pub mod io;
pub mod join;
pub mod mapreduce;
pub mod net;
pub mod ops;
pub mod partition;

pub mod row;
pub mod scalar;
pub mod simd;
pub mod table;
pub mod util;

#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export commonly used types
pub use crate::ctx::CylonContext;
pub use crate::data_types::{DataType, Layout, Type};
pub use crate::error::{CylonError, CylonResult};
pub use crate::table::Table;

/// The main entry point and version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
