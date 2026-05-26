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

//! Logging utilities
//!
//! Ported from cpp/src/cylon/util/logging.hpp
//! Uses standard Rust log crate instead of glog

use log::{debug, error, info, trace, warn};

/// Initialize logging with default configuration.
/// Writes to stdout so Lambda Node.js runtime captures it in CloudWatch.
/// Colors are disabled (WriteStyle::Never) for clean log output.
pub fn init_logging() {
    env_logger::Builder::from_default_env()
        .write_style(env_logger::WriteStyle::Never)
        .target(env_logger::Target::Stdout)
        .init();
}

/// Initialize logging with specific level.
/// Writes to stdout so Lambda Node.js runtime captures it in CloudWatch.
pub fn init_logging_with_level(level: log::LevelFilter) {
    env_logger::Builder::from_default_env()
        .filter_level(level)
        .write_style(env_logger::WriteStyle::Never)
        .target(env_logger::Target::Stdout)
        .init();
}

/// Log macros that mirror the C++ interface
#[macro_export]
macro_rules! cylon_info {
    ($($arg:tt)*) => {
        log::info!($($arg)*)
    };
}

#[macro_export]
macro_rules! cylon_debug {
    ($($arg:tt)*) => {
        log::debug!($($arg)*)
    };
}

#[macro_export]
macro_rules! cylon_warn {
    ($($arg:tt)*) => {
        log::warn!($($arg)*)
    };
}

#[macro_export]
macro_rules! cylon_error {
    ($($arg:tt)*) => {
        log::error!($($arg)*)
    };
}

#[macro_export]
macro_rules! cylon_trace {
    ($($arg:tt)*) => {
        log::trace!($($arg)*)
    };
}