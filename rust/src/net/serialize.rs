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

//! Arrow table and column serialization for distributed operations
//!
//! This module provides serialization/deserialization of Arrow tables and columns
//! for efficient network transmission in distributed operations.
//!
//! Key Features:
//! - Table serialization using Arrow IPC (Inter-Process Communication) format
//! - Column serialization using raw buffer extraction for collective operations
//! - Zero-copy where possible
//! - Maintains columnar format during transmission
//!
//! Ported from cpp/src/cylon/net/serialize.hpp and cpp/src/cylon/serialize/table_serialize.hpp

use std::sync::Arc;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow::ipc::CompressionType;
use arrow::record_batch::RecordBatch;
use arrow::array::ArrayRef;
use arrow::datatypes::DataType as ArrowDataType;

use crate::error::{Code, CylonError, CylonResult};
use crate::ctx::CylonContext;
use crate::table::Table;
use crate::arrow::arrow_types::to_cylon_type_id;

/// Compression algorithm for Arrow IPC serialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IpcCompression {
    /// No compression
    #[default]
    None,
    /// LZ4 frame compression (fast)
    Lz4,
    /// Zstandard compression (better ratio)
    Zstd,
}

impl IpcCompression {
    /// Convert to Arrow IPC CompressionType
    fn to_arrow_compression(self) -> Option<CompressionType> {
        match self {
            IpcCompression::None => None,
            IpcCompression::Lz4 => Some(CompressionType::LZ4_FRAME),
            IpcCompression::Zstd => Some(CompressionType::ZSTD),
        }
    }
}

/// Serialize an Arrow RecordBatch to bytes using Arrow IPC format
///
/// # Arguments
/// * `batch` - The RecordBatch to serialize
///
/// # Returns
/// A vector of bytes containing the serialized data in Arrow IPC stream format
///
/// # Example
/// ```ignore
/// let batch = // ... create RecordBatch
/// let bytes = serialize_record_batch(&batch)?;
/// // Send bytes over network
/// ```
pub fn serialize_record_batch(batch: &RecordBatch) -> CylonResult<Vec<u8>> {
    let mut buffer = Vec::new();

    {
        // Create a StreamWriter that writes to our buffer
        let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema())
            .map_err(|e| CylonError::new(
                Code::Invalid,
                format!("Failed to create IPC writer: {}", e)
            ))?;

        // Write the batch
        writer.write(batch)
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to write batch: {}", e)
            ))?;

        // Finish writing (flushes footer)
        writer.finish()
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to finish writing: {}", e)
            ))?;
    }

    Ok(buffer)
}

/// Deserialize bytes to an Arrow RecordBatch using Arrow IPC format
///
/// # Arguments
/// * `data` - Bytes containing serialized RecordBatch in Arrow IPC stream format
///
/// # Returns
/// The deserialized RecordBatch
///
/// # Example
/// ```ignore
/// let bytes = // ... receive from network
/// let batch = deserialize_record_batch(&bytes)?;
/// ```
pub fn deserialize_record_batch(data: &[u8]) -> CylonResult<RecordBatch> {
    // Create a StreamReader from the bytes
    let cursor = std::io::Cursor::new(data);
    let mut reader = StreamReader::try_new(cursor, None)
        .map_err(|e| CylonError::new(
            Code::Invalid,
            format!("Failed to create IPC reader: {}", e)
        ))?;

    // Read the first (and should be only) batch
    reader.next()
        .ok_or_else(|| CylonError::new(
            Code::Invalid,
            "No batch found in serialized data".to_string()
        ))?
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to read batch: {}", e)
        ))
}

/// Serialize a Cylon Table to bytes using Arrow IPC format
///
/// This function serializes all batches in the table along with the schema.
/// The format is compatible with Arrow IPC stream format.
///
/// # Arguments
/// * `table` - The Table to serialize
///
/// # Returns
/// A vector of bytes containing the serialized table
///
/// # Example
/// ```ignore
/// let table = // ... create Table
/// let bytes = serialize_table(&table)?;
/// // Send bytes over network
/// ```
pub fn serialize_table(table: &Table) -> CylonResult<Vec<u8>> {
    serialize_table_with_compression(table, IpcCompression::None)
}

/// Serialize a Cylon Table to bytes using Arrow IPC format with compression.
///
/// This function serializes all batches in the table with optional compression.
/// Compression is applied at the Arrow buffer level for efficient storage.
///
/// # Arguments
/// * `table` - The Table to serialize
/// * `compression` - Compression algorithm to use
///
/// # Returns
/// A vector of bytes containing the serialized (and optionally compressed) table
///
/// # Example
/// ```ignore
/// let table = // ... create Table
/// let bytes = serialize_table_with_compression(&table, IpcCompression::Lz4)?;
/// ```
pub fn serialize_table_with_compression(
    table: &Table,
    compression: IpcCompression,
) -> CylonResult<Vec<u8>> {
    let mut buffer = Vec::new();

    // Get the table's schema
    let schema = table.schema().ok_or_else(|| CylonError::new(
        Code::Invalid,
        "Table has no schema".to_string()
    ))?;

    {
        // Create IPC write options with compression if specified
        let write_options = IpcWriteOptions::default()
            .try_with_compression(compression.to_arrow_compression())
            .map_err(|e| CylonError::new(
                Code::Invalid,
                format!("Failed to set compression options: {}", e)
            ))?;

        // Create a StreamWriter with the table's schema and options
        let mut writer = StreamWriter::try_new_with_options(&mut buffer, &schema, write_options)
            .map_err(|e| CylonError::new(
                Code::Invalid,
                format!("Failed to create IPC writer: {}", e)
            ))?;

        // Write all batches
        for i in 0..table.num_batches() {
            if let Some(batch) = table.batch(i) {
                writer.write(&batch)
                    .map_err(|e| CylonError::new(
                        Code::IoError,
                        format!("Failed to write batch {}: {}", i, e)
                    ))?;
            }
        }

        // Finish writing
        writer.finish()
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to finish writing: {}", e)
            ))?;
    }

    Ok(buffer)
}

/// Deserialize bytes to a Cylon Table using Arrow IPC format
///
/// This function deserializes all batches from the Arrow IPC stream format
/// and reconstructs the Table.
///
/// # Arguments
/// * `ctx` - CylonContext for the table
/// * `data` - Bytes containing serialized Table in Arrow IPC stream format
///
/// # Returns
/// The deserialized Table
///
/// # Example
/// ```ignore
/// let ctx = CylonContext::new()?;
/// let bytes = // ... receive from network
/// let table = deserialize_table(ctx, &bytes)?;
/// ```
pub fn deserialize_table(ctx: Arc<CylonContext>, data: &[u8]) -> CylonResult<Table> {
    // Create a StreamReader from the bytes
    let cursor = std::io::Cursor::new(data);
    let mut reader = StreamReader::try_new(cursor, None)
        .map_err(|e| CylonError::new(
            Code::Invalid,
            format!("Failed to create IPC reader: {}", e)
        ))?;

    // Read all batches
    let mut batches = Vec::new();

    for result in reader {
        let batch = result.map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to read batch: {}", e)
        ))?;
        batches.push(batch);
    }

    // Create Table from batches (handles empty case automatically)
    Table::from_record_batches(ctx, batches)
}

// =============================================================================
// Column Serialization for Collective Operations
// =============================================================================
//
// Ported from cpp/src/cylon/serialize/table_serialize.hpp and table_serialize.cpp
//
// This section provides raw buffer extraction from Arrow arrays for use in
// collective operations like AllReduce and AllGather. Unlike IPC serialization,
// this extracts the raw underlying buffers directly.
//
// Buffer layout per column: [validity, offsets, data] - 3 buffers

/// Check if a type is fixed-width (primitive types)
fn is_fixed_width(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Boolean
            | ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
            | ArrowDataType::Float16
            | ArrowDataType::Float32
            | ArrowDataType::Float64
            | ArrowDataType::Date32
            | ArrowDataType::Date64
            | ArrowDataType::Time32(_)
            | ArrowDataType::Time64(_)
            | ArrowDataType::Timestamp(_, _)
            | ArrowDataType::Duration(_)
            | ArrowDataType::Interval(_)
            | ArrowDataType::FixedSizeBinary(_)
            | ArrowDataType::Decimal128(_, _)
            | ArrowDataType::Decimal256(_, _)
    )
}

/// Check if a type is binary-like (variable-length with i32 offsets)
fn is_binary_like(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::Utf8 | ArrowDataType::Binary
    )
}

/// Check if a type is large binary-like (variable-length with i64 offsets)
fn is_large_binary_like(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type,
        ArrowDataType::LargeUtf8 | ArrowDataType::LargeBinary
    )
}

/// Get the byte width of a fixed-width type
fn get_byte_width(data_type: &ArrowDataType) -> Option<usize> {
    match data_type {
        ArrowDataType::Boolean => Some(0), // Boolean is special, uses bit packing
        ArrowDataType::Int8 | ArrowDataType::UInt8 => Some(1),
        ArrowDataType::Int16 | ArrowDataType::UInt16 | ArrowDataType::Float16 => Some(2),
        ArrowDataType::Int32 | ArrowDataType::UInt32 | ArrowDataType::Float32 |
        ArrowDataType::Date32 | ArrowDataType::Time32(_) => Some(4),
        ArrowDataType::Int64 | ArrowDataType::UInt64 | ArrowDataType::Float64 |
        ArrowDataType::Date64 | ArrowDataType::Time64(_) |
        ArrowDataType::Timestamp(_, _) | ArrowDataType::Duration(_) => Some(8),
        ArrowDataType::Interval(_) => Some(16),
        ArrowDataType::FixedSizeBinary(size) => Some(*size as usize),
        ArrowDataType::Decimal128(_, _) => Some(16),
        ArrowDataType::Decimal256(_, _) => Some(32),
        _ => None,
    }
}

/// Bytes needed for a given number of bits
fn bytes_for_bits(bits: usize) -> usize {
    (bits + 7) / 8
}

/// Collect bitmap info from ArrayData
///
/// Corresponds to C++ CollectBitmapInfo<buf_idx> template function
/// (table_serialize.cpp:26-52)
///
/// # Arguments
/// * `data` - ArrayData to extract bitmap from
/// * `buf_idx` - Buffer index (0 for validity bitmap, 1 for boolean data)
///
/// # Returns
/// * (buffer_size, buffer_data) - Size and copied data of the buffer
fn collect_bitmap_info(
    data: &arrow::array::ArrayData,
    buf_idx: usize,
) -> CylonResult<(i32, Option<Vec<u8>>)> {
    // For validity bitmap (buf_idx == 0), skip if no nulls
    if buf_idx == 0 && data.nulls().is_none() {
        return Ok((0, None));
    }

    // Calculate buffer size in bytes
    let buffer_size = bytes_for_bits(data.len()) as i32;

    let offset = data.offset();
    if let Some(buffer) = data.buffers().get(buf_idx) {
        if offset == 0 {
            // No offset - copy buffer directly
            let slice = &buffer.as_slice()[..buffer_size as usize];
            return Ok((buffer_size, Some(slice.to_vec())));
        } else if offset % 8 == 0 {
            // Offset is at byte boundary
            let byte_offset = offset / 8;
            let slice = &buffer.as_slice()[byte_offset..byte_offset + buffer_size as usize];
            return Ok((buffer_size, Some(slice.to_vec())));
        } else {
            // Non-byte boundary offset - need to copy and realign bitmap
            let mut new_buffer = vec![0u8; buffer_size as usize];
            for i in 0..data.len() {
                let src_bit_idx = offset + i;
                let src_byte = src_bit_idx / 8;
                let src_bit = src_bit_idx % 8;
                let bit_val = (buffer.as_slice()[src_byte] >> src_bit) & 1;

                let dst_byte = i / 8;
                let dst_bit = i % 8;
                new_buffer[dst_byte] |= bit_val << dst_bit;
            }
            return Ok((buffer_size, Some(new_buffer)));
        }
    } else if buf_idx == 0 {
        // Validity buffer doesn't exist but we have nulls - shouldn't happen
        return Ok((0, None));
    } else {
        return Err(CylonError::new(
            Code::Invalid,
            format!("Buffer {} not found in ArrayData", buf_idx),
        ));
    }
}

/// Collect offset buffer from ArrayData
///
/// Corresponds to C++ CollectOffsetBuffer function (table_serialize.cpp:89-112)
///
/// # Arguments
/// * `data` - ArrayData to extract offsets from
///
/// # Returns
/// * (buffer_size, buffer_data) - Size and copied data of the offset buffer
fn collect_offset_buffer(
    data: &arrow::array::ArrayData,
) -> CylonResult<(i32, Option<Vec<u8>>)> {
    let data_type = data.data_type();

    if is_fixed_width(data_type) {
        return Ok((0, None));
    }

    if is_binary_like(data_type) {
        // For binary-like types, offset buffer is at index 0
        // (length + 1) offsets of i32
        let buffer_size = ((data.len() + 1) * std::mem::size_of::<i32>()) as i32;
        if let Some(buffer) = data.buffers().get(0) {
            let offset_start = data.offset() * std::mem::size_of::<i32>();
            let slice = &buffer.as_slice()[offset_start..offset_start + buffer_size as usize];
            return Ok((buffer_size, Some(slice.to_vec())));
        }
        return Ok((buffer_size, None));
    }

    if is_large_binary_like(data_type) {
        // For large binary-like types, offset buffer is at index 0
        // (length + 1) offsets of i64
        let buffer_size = ((data.len() + 1) * std::mem::size_of::<i64>()) as i32;
        if let Some(buffer) = data.buffers().get(0) {
            let offset_start = data.offset() * std::mem::size_of::<i64>();
            let slice = &buffer.as_slice()[offset_start..offset_start + buffer_size as usize];
            return Ok((buffer_size, Some(slice.to_vec())));
        }
        return Ok((buffer_size, None));
    }

    Err(CylonError::new(
        Code::Invalid,
        format!("Unsupported offset type for serialization: {:?}", data_type),
    ))
}

/// Collect data buffer from ArrayData
///
/// Corresponds to C++ CollectDataBuffer function (table_serialize.cpp:54-87)
///
/// # Arguments
/// * `data` - ArrayData to extract data from
///
/// # Returns
/// * (buffer_size, buffer_data) - Size and copied data of the data buffer
fn collect_data_buffer(
    data: &arrow::array::ArrayData,
) -> CylonResult<(i32, Option<Vec<u8>>)> {
    let data_type = data.data_type();

    // Boolean is stored as a bitmap, call CollectBitmapInfo with buf_idx=1
    if matches!(data_type, ArrowDataType::Boolean) {
        return collect_bitmap_info(data, 1);
    }

    if is_fixed_width(data_type) {
        if let Some(byte_width) = get_byte_width(data_type) {
            let buffer_size = (byte_width * data.len()) as i32;
            if let Some(buffer) = data.buffers().get(0) {
                let offset_bytes = data.offset() * byte_width;
                let slice = &buffer.as_slice()[offset_bytes..offset_bytes + buffer_size as usize];
                return Ok((buffer_size, Some(slice.to_vec())));
            }
            return Ok((buffer_size, None));
        }
        return Ok((0, None));
    }

    if is_binary_like(data_type) {
        // Get start and end offset from offset buffer
        if let Some(offset_buffer) = data.buffers().get(0) {
            let offsets: &[i32] = unsafe {
                std::slice::from_raw_parts(
                    offset_buffer.as_ptr() as *const i32,
                    offset_buffer.len() / std::mem::size_of::<i32>(),
                )
            };
            let start_idx = data.offset();
            let start_offset = offsets[start_idx] as usize;
            let end_offset = offsets[start_idx + data.len()] as usize;

            let buffer_size = (end_offset - start_offset) as i32;
            if let Some(data_buf) = data.buffers().get(1) {
                let slice = &data_buf.as_slice()[start_offset..end_offset];
                return Ok((buffer_size, Some(slice.to_vec())));
            }
            return Ok((buffer_size, None));
        }
        return Ok((0, None));
    }

    if is_large_binary_like(data_type) {
        // Get start and end offset from offset buffer (i64)
        if let Some(offset_buffer) = data.buffers().get(0) {
            let offsets: &[i64] = unsafe {
                std::slice::from_raw_parts(
                    offset_buffer.as_ptr() as *const i64,
                    offset_buffer.len() / std::mem::size_of::<i64>(),
                )
            };
            let start_idx = data.offset();
            let start_offset = offsets[start_idx] as usize;
            let end_offset = offsets[start_idx + data.len()] as usize;

            let buffer_size = (end_offset - start_offset) as i32;
            if let Some(data_buf) = data.buffers().get(1) {
                let slice = &data_buf.as_slice()[start_offset..end_offset];
                return Ok((buffer_size, Some(slice.to_vec())));
            }
            return Ok((buffer_size, None));
        }
        return Ok((0, None));
    }

    Err(CylonError::new(
        Code::Invalid,
        format!("Unsupported data type for serialization: {:?}", data_type),
    ))
}

/// Trait for column serialization
///
/// Corresponds to C++ ColumnSerializer interface (table_serialize.hpp:69-77)
pub trait ColumnSerializer {
    /// Get the sizes of the three buffers [validity, offsets, data]
    fn buffer_sizes(&self) -> [i32; 3];

    /// Get references to the three data buffers
    fn data_buffers(&self) -> [Option<&[u8]>; 3];

    /// Get the Cylon data type ID
    fn get_data_type_id(&self) -> i32;
}

/// Cylon column serializer implementation
///
/// Corresponds to C++ CylonColumnSerializer class (table_serialize.hpp:79-104)
pub struct CylonColumnSerializer {
    /// The underlying Arrow array (kept for lifetime)
    array_: ArrayRef,
    /// Data buffer contents [validity, offsets, data]
    data_bufs_: [Option<Vec<u8>>; 3],
    /// Buffer sizes [validity, offsets, data]
    buf_sizes_: [i32; 3],
}

impl CylonColumnSerializer {
    /// Create a new column serializer from an Arrow array
    ///
    /// Corresponds to C++ CylonColumnSerializer::Make (table_serialize.cpp:401-424)
    pub fn make(array: &ArrayRef) -> CylonResult<Self> {
        let mut buffer_sizes = [0i32; 3];
        let mut data_buffers: [Option<Vec<u8>>; 3] = [None, None, None];

        if array.len() > 0 {
            let data = array.to_data();

            // Order: validity, offsets, data
            let (size0, buf0) = collect_bitmap_info(&data, 0)?;
            buffer_sizes[0] = size0;
            data_buffers[0] = buf0;

            let (size1, buf1) = collect_offset_buffer(&data)?;
            buffer_sizes[1] = size1;
            data_buffers[1] = buf1;

            let (size2, buf2) = collect_data_buffer(&data)?;
            buffer_sizes[2] = size2;
            data_buffers[2] = buf2;
        }

        Ok(Self {
            array_: array.clone(),
            data_bufs_: data_buffers,
            buf_sizes_: buffer_sizes,
        })
    }
}

impl ColumnSerializer for CylonColumnSerializer {
    fn buffer_sizes(&self) -> [i32; 3] {
        self.buf_sizes_
    }

    fn data_buffers(&self) -> [Option<&[u8]>; 3] {
        [
            self.data_bufs_[0].as_ref().map(|v| v.as_slice()),
            self.data_bufs_[1].as_ref().map(|v| v.as_slice()),
            self.data_bufs_[2].as_ref().map(|v| v.as_slice()),
        ]
    }

    fn get_data_type_id(&self) -> i32 {
        to_cylon_type_id(self.array_.data_type()) as i32
    }
}
