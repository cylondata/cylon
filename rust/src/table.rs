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

//! Table - main data structure for Cylon
//!
//! Ported from cpp/src/cylon/table.hpp

use std::sync::Arc;
use arrow::array::RecordBatch;
use arrow::datatypes::Schema;

use crate::ctx::CylonContext;
use crate::error::CylonResult;
use crate::indexing::BaseArrowIndex;
use crate::net::serialize::{deserialize_record_batch, serialize_record_batch};

pub mod column;
pub use column::{Column, FromVector};

/// Table provides the main API for using cylon for data processing
/// Corresponds to C++ Table class from cpp/src/cylon/table.hpp
#[derive(Clone)]
pub struct Table {
    ctx: Arc<CylonContext>,
    // Using Arrow RecordBatch internally (similar to C++ using arrow::Table)
    batches: Vec<RecordBatch>,
    retain: bool,
    // Index for the table (C++ table.hpp:179)
    base_arrow_index: Option<Arc<dyn BaseArrowIndex>>,
}

impl Table {
    /// Create a table from Arrow RecordBatch
    pub fn from_record_batch(
        ctx: Arc<CylonContext>,
        batch: RecordBatch,
    ) -> CylonResult<Self> {
        Ok(Self {
            ctx,
            batches: vec![batch],
            retain: true,
            base_arrow_index: None,
        })
    }

    /// Create a table from multiple Arrow RecordBatches
    pub fn from_record_batches(
        ctx: Arc<CylonContext>,
        batches: Vec<RecordBatch>,
    ) -> CylonResult<Self> {
        Ok(Self {
            ctx,
            batches,
            retain: true,
            base_arrow_index: None,
        })
    }

    /// Get the number of columns
    pub fn columns(&self) -> i32 {
        if let Some(batch) = self.batches.first() {
            batch.num_columns() as i32
        } else {
            0
        }
    }

    /// Get the number of rows
    pub fn rows(&self) -> i64 {
        self.batches.iter().map(|b| b.num_rows() as i64).sum()
    }

    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.rows() == 0
    }

    /// Get the context
    pub fn get_context(&self) -> Arc<CylonContext> {
        self.ctx.clone()
    }

    /// Get the schema
    pub fn schema(&self) -> Option<Arc<Schema>> {
        self.batches.first().map(|b| b.schema())
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        if let Some(schema) = self.schema() {
            schema.fields().iter().map(|f| f.name().clone()).collect()
        } else {
            Vec::new()
        }
    }

    /// Set retention flag
    pub fn retain_memory(&mut self, retain: bool) {
        self.retain = retain;
    }

    /// Check if table retains memory
    pub fn is_retain(&self) -> bool {
        self.retain
    }

    /// Get the number of batches in the table
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }

    /// Get a reference to a specific batch
    pub fn batch(&self, index: usize) -> Option<&RecordBatch> {
        self.batches.get(index)
    }

    /// Get all batches
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// Get a column as a concatenated array (across all batches)
    /// Corresponds to C++ get_table()->column(idx)
    pub fn column(&self, col_idx: usize) -> CylonResult<arrow::array::ArrayRef> {
        if self.batches.is_empty() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "Table has no batches".to_string(),
            ));
        }

        if col_idx >= self.batches[0].num_columns() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                format!("Column index {} out of range", col_idx),
            ));
        }

        // Single batch - return directly
        if self.batches.len() == 1 {
            return Ok(self.batches[0].column(col_idx).clone());
        }

        // Multiple batches - concatenate
        let arrays: Vec<&dyn arrow::array::Array> = self.batches
            .iter()
            .map(|b| b.column(col_idx).as_ref())
            .collect();

        arrow::compute::concat(&arrays)
            .map_err(|e| crate::error::CylonError::from(e))
    }

    /// Read a table from a CSV file
    /// Corresponds to C++ FromCSV
    pub fn from_csv(
        ctx: Arc<CylonContext>,
        path: &str,
        options: &crate::io::CsvReadOptions,
    ) -> CylonResult<Self> {
        crate::io::read_csv(ctx, path, options)
    }

    /// Read a table from a CSV file with default options
    pub fn from_csv_default(
        ctx: Arc<CylonContext>,
        path: &str,
    ) -> CylonResult<Self> {
        crate::io::read_csv(ctx, path, &crate::io::CsvReadOptions::default())
    }

    /// Write the table to a CSV file
    /// Corresponds to C++ WriteCSV
    pub fn to_csv(
        &self,
        path: &str,
        options: &crate::io::CsvWriteOptions,
    ) -> CylonResult<()> {
        crate::io::write_csv(self, path, options)
    }

    /// Write the table to a CSV file with default options
    pub fn to_csv_default(&self, path: &str) -> CylonResult<()> {
        crate::io::write_csv(self, path, &crate::io::CsvWriteOptions::default())
    }

    /// Read a table from a Parquet file
    /// Corresponds to C++ ReadParquet (arrow_io.cpp:61-86)
    #[cfg(feature = "parquet")]
    pub fn from_parquet(
        ctx: Arc<CylonContext>,
        path: &str,
    ) -> CylonResult<Self> {
        crate::io::read_parquet(ctx, path)
    }

    /// Write the table to a Parquet file
    /// Corresponds to C++ WriteParquet (arrow_io.cpp:89-110)
    #[cfg(feature = "parquet")]
    pub fn to_parquet(
        &self,
        path: &str,
        options: crate::io::ParquetOptions,
    ) -> CylonResult<()> {
        crate::io::write_parquet(self.ctx.clone(), self, path, options)
    }

    /// Write the table to a Parquet file with default options
    #[cfg(feature = "parquet")]
    pub fn to_parquet_default(&self, path: &str) -> CylonResult<()> {
        crate::io::write_parquet(self.ctx.clone(), self, path, crate::io::ParquetOptions::default())
    }

    /// Project (select) specific columns from the table
    /// Corresponds to C++ Project function (table.cpp:1212)
    ///
    /// # Arguments
    /// * `column_indices` - Indices of columns to include in the projection
    ///
    /// # Returns
    /// A new table with only the specified columns
    pub fn project(&self, column_indices: &[usize]) -> CylonResult<Table> {
        if column_indices.is_empty() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "column_indices cannot be empty".to_string(),
            ));
        }

        let mut projected_batches = Vec::new();

        for batch in &self.batches {
            // Validate indices
            for &idx in column_indices {
                if idx >= batch.num_columns() {
                    return Err(crate::error::CylonError::new(
                        crate::error::Code::Invalid,
                        format!("Column index {} out of range (table has {} columns)",
                                idx, batch.num_columns()),
                    ));
                }
            }

            // Build new schema with selected columns
            let fields: Vec<_> = column_indices.iter()
                .map(|&idx| batch.schema().field(idx).clone())
                .collect();
            let new_schema = Arc::new(Schema::new(fields));

            // Build column arrays
            let columns: Vec<_> = column_indices.iter()
                .map(|&idx| batch.column(idx).clone())
                .collect();

            // Create new batch
            let projected_batch = RecordBatch::try_new(new_schema, columns)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to create projected batch: {}", e),
                ))?;

            projected_batches.push(projected_batch);
        }

        Table::from_record_batches(self.ctx.clone(), projected_batches)
    }

    /// Project by column names instead of indices
    ///
    /// # Arguments
    /// * `column_names` - Names of columns to include in the projection
    ///
    /// # Returns
    /// A new table with only the specified columns
    pub fn project_by_names(&self, column_names: &[&str]) -> CylonResult<Table> {
        if column_names.is_empty() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "column_names cannot be empty".to_string(),
            ));
        }

        // Get schema from first batch to resolve column names
        let schema = self.schema().ok_or_else(|| {
            crate::error::CylonError::new(crate::error::Code::Invalid, "Table has no schema".to_string())
        })?;

        // Resolve column names to indices
        let mut indices = Vec::new();
        for &col_name in column_names {
            match schema.index_of(col_name) {
                Ok(idx) => indices.push(idx),
                Err(_) => {
                    return Err(crate::error::CylonError::new(
                        crate::error::Code::Invalid,
                        format!("Column '{}' not found in table", col_name),
                    ));
                }
            }
        }

        self.project(&indices)
    }

    /// Slice the table to get a subset of rows
    /// Corresponds to C++ Slice function
    ///
    /// # Arguments
    /// * `offset` - Starting row index (0-based)
    /// * `length` - Number of rows to include
    ///
    /// # Returns
    /// A new table with the specified row range
    pub fn slice(&self, offset: usize, length: usize) -> CylonResult<Table> {
        use arrow::compute::concat_batches;

        let total_rows = self.rows() as usize;

        // Handle empty table - return as-is (matches C++ behavior at slice.cpp:39-42)
        if total_rows == 0 {
            return Table::from_record_batches(self.ctx.clone(), self.batches.clone());
        }

        // Allow offset == total_rows (returns empty slice, matches C++ test at slice_test.cpp:128)
        // But offset > total_rows is invalid
        if offset > total_rows {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                format!("Offset {} is out of range (table has {} rows)", offset, total_rows),
            ));
        }

        // Combine all batches first if we have multiple
        let combined = if self.batches.len() > 1 {
            let schema = self.schema().ok_or_else(|| {
                crate::error::CylonError::new(crate::error::Code::Invalid, "Table has no schema".to_string())
            })?;

            concat_batches(&schema, &self.batches)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to combine batches: {}", e),
                ))?
        } else if self.batches.len() == 1 {
            self.batches[0].clone()
        } else {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "Table has no batches".to_string(),
            ));
        };

        // Adjust length if it exceeds table bounds
        let actual_length = length.min(total_rows - offset);

        // Slice the combined batch
        let sliced = combined.slice(offset, actual_length);

        Table::from_record_batch(self.ctx.clone(), sliced)
    }

    /// Get the first n rows of the table
    /// Corresponds to C++ Head function
    ///
    /// # Arguments
    /// * `n` - Number of rows to return
    ///
    /// # Returns
    /// A new table with the first n rows
    pub fn head(&self, n: usize) -> CylonResult<Table> {
        self.slice(0, n)
    }

    /// Get the last n rows of the table
    /// Corresponds to C++ Tail function
    ///
    /// # Arguments
    /// * `n` - Number of rows to return
    ///
    /// # Returns
    /// A new table with the last n rows
    pub fn tail(&self, n: usize) -> CylonResult<Table> {
        let total_rows = self.rows() as usize;
        if n >= total_rows {
            // Return the whole table if n >= total rows
            self.slice(0, total_rows)
        } else {
            self.slice(total_rows - n, n)
        }
    }

    /// Merge multiple tables vertically (concatenate rows)
    /// Corresponds to C++ Merge function (table.cpp:343)
    ///
    /// All tables must have the same schema
    ///
    /// # Arguments
    /// * `tables` - Vector of tables to merge with this table
    ///
    /// # Returns
    /// A new table containing all rows from all tables
    pub fn merge(&self, tables: &[&Table]) -> CylonResult<Table> {
        // Collect all batches from all tables
        let mut all_batches = self.batches.clone();

        let schema = self.schema().ok_or_else(|| {
            crate::error::CylonError::new(crate::error::Code::Invalid, "Table has no schema".to_string())
        })?;

        // Validate all tables have same schema and collect their batches
        for table in tables {
            let other_schema = table.schema().ok_or_else(|| {
                crate::error::CylonError::new(crate::error::Code::Invalid, "Table to merge has no schema".to_string())
            })?;

            if !schema.eq(&other_schema) {
                return Err(crate::error::CylonError::new(
                    crate::error::Code::Invalid,
                    "Cannot merge tables with different schemas".to_string(),
                ));
            }

            all_batches.extend(table.batches.clone());
        }

        Table::from_record_batches(self.ctx.clone(), all_batches)
    }

    /// Sort table by a single column
    /// Corresponds to C++ Sort function (table.cpp:372)
    ///
    /// # Arguments
    /// * `sort_column` - Index of column to sort by
    /// * `ascending` - true for ascending, false for descending
    ///
    /// # Returns
    /// A new sorted table
    pub fn sort(&self, sort_column: usize, ascending: bool) -> CylonResult<Table> {
        use arrow::compute::{lexsort_to_indices, take, SortColumn};

        // If table has 0 or 1 rows, no need to sort
        if self.rows() < 2 {
            return Table::from_record_batches(self.ctx.clone(), self.batches.clone());
        }

        // Combine all batches first
        let combined = if self.batches.len() > 1 {
            let schema = self.schema().ok_or_else(|| {
                crate::error::CylonError::new(crate::error::Code::Invalid, "Table has no schema".to_string())
            })?;

            arrow::compute::concat_batches(&schema, &self.batches)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to combine batches: {}", e),
                ))?
        } else if self.batches.len() == 1 {
            self.batches[0].clone()
        } else {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "Table has no batches".to_string(),
            ));
        };

        // Validate column index
        if sort_column >= combined.num_columns() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                format!("Sort column index {} out of range (table has {} columns)",
                        sort_column, combined.num_columns()),
            ));
        }

        // Create sort column specification
        let sort_cols = vec![SortColumn {
            values: combined.column(sort_column).clone(),
            options: Some(arrow::compute::SortOptions {
                descending: !ascending,
                nulls_first: false,
            }),
        }];

        // Get sorted indices
        let indices = lexsort_to_indices(&sort_cols, None)
            .map_err(|e| crate::error::CylonError::new(
                crate::error::Code::ExecutionError,
                format!("Failed to compute sort indices: {}", e),
            ))?;

        // Apply indices to all columns
        let mut sorted_columns = Vec::new();
        for i in 0..combined.num_columns() {
            let sorted_col = take(combined.column(i), &indices, None)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to reorder column {}: {}", i, e),
                ))?;
            sorted_columns.push(sorted_col);
        }

        // Create sorted batch
        let sorted_batch = RecordBatch::try_new(combined.schema(), sorted_columns)
            .map_err(|e| crate::error::CylonError::new(
                crate::error::Code::ExecutionError,
                format!("Failed to create sorted batch: {}", e),
            ))?;

        Table::from_record_batch(self.ctx.clone(), sorted_batch)
    }

    /// Sort table by multiple columns
    /// Corresponds to C++ Sort function (table.cpp:395)
    ///
    /// # Arguments
    /// * `sort_columns` - Indices of columns to sort by (in order of priority)
    /// * `sort_directions` - Sort direction for each column (true = ascending, false = descending)
    ///
    /// # Returns
    /// A new sorted table
    pub fn sort_multi(&self, sort_columns: &[usize], sort_directions: &[bool]) -> CylonResult<Table> {
        use arrow::compute::{lexsort_to_indices, take, SortColumn};

        if sort_columns.is_empty() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "sort_columns cannot be empty".to_string(),
            ));
        }

        if sort_columns.len() != sort_directions.len() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "sort_columns and sort_directions must have the same length".to_string(),
            ));
        }

        // Single column sort - delegate to simpler method
        if sort_columns.len() == 1 {
            return self.sort(sort_columns[0], sort_directions[0]);
        }

        // If table has 0 or 1 rows, no need to sort
        if self.rows() < 2 {
            return Table::from_record_batches(self.ctx.clone(), self.batches.clone());
        }

        // Combine all batches first
        let combined = if self.batches.len() > 1 {
            let schema = self.schema().ok_or_else(|| {
                crate::error::CylonError::new(crate::error::Code::Invalid, "Table has no schema".to_string())
            })?;

            arrow::compute::concat_batches(&schema, &self.batches)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to combine batches: {}", e),
                ))?
        } else if self.batches.len() == 1 {
            self.batches[0].clone()
        } else {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "Table has no batches".to_string(),
            ));
        };

        // Validate column indices and build sort column specs
        let mut sort_cols = Vec::new();
        for (&col_idx, &ascending) in sort_columns.iter().zip(sort_directions.iter()) {
            if col_idx >= combined.num_columns() {
                return Err(crate::error::CylonError::new(
                    crate::error::Code::Invalid,
                    format!("Sort column index {} out of range (table has {} columns)",
                            col_idx, combined.num_columns()),
                ));
            }

            sort_cols.push(SortColumn {
                values: combined.column(col_idx).clone(),
                options: Some(arrow::compute::SortOptions {
                    descending: !ascending,
                    nulls_first: false,
                }),
            });
        }

        // Get sorted indices
        let indices = lexsort_to_indices(&sort_cols, None)
            .map_err(|e| crate::error::CylonError::new(
                crate::error::Code::ExecutionError,
                format!("Failed to compute sort indices: {}", e),
            ))?;

        // Apply indices to all columns
        let mut sorted_columns = Vec::new();
        for i in 0..combined.num_columns() {
            let sorted_col = take(combined.column(i), &indices, None)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to reorder column {}: {}", i, e),
                ))?;
            sorted_columns.push(sorted_col);
        }

        // Create sorted batch
        let sorted_batch = RecordBatch::try_new(combined.schema(), sorted_columns)
            .map_err(|e| crate::error::CylonError::new(
                crate::error::Code::ExecutionError,
                format!("Failed to create sorted batch: {}", e),
            ))?;

        Table::from_record_batch(self.ctx.clone(), sorted_batch)
    }

    /// Filter rows based on a boolean mask array
    /// Corresponds to C++ Select function (table.cpp:892)
    ///
    /// This accepts a pre-computed boolean array indicating which rows to keep.
    ///
    /// # Arguments
    /// * `mask` - Boolean array where true means keep the row, false means discard
    ///
    /// # Returns
    /// A new table containing only the rows where mask is true
    pub fn select(&self, mask: &arrow::array::BooleanArray) -> CylonResult<Table> {
        use arrow::compute::filter_record_batch;

        // Validate mask length
        if mask.len() != self.rows() as usize {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                format!("Mask length {} does not match table rows {}",
                        mask.len(), self.rows()),
            ));
        }

        // Filter each batch
        let mut filtered_batches = Vec::new();

        if self.batches.len() == 1 {
            // Single batch - simple case
            let filtered = filter_record_batch(&self.batches[0], mask)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to filter batch: {}", e),
                ))?;
            filtered_batches.push(filtered);
        } else {
            // Multiple batches - need to split mask accordingly
            let mut mask_offset = 0;
            for batch in &self.batches {
                let batch_len = batch.num_rows();

                // Slice the mask for this batch
                let batch_mask = mask.slice(mask_offset, batch_len);

                let filtered = filter_record_batch(batch, &batch_mask)
                    .map_err(|e| crate::error::CylonError::new(
                        crate::error::Code::ExecutionError,
                        format!("Failed to filter batch: {}", e),
                    ))?;

                if filtered.num_rows() > 0 {
                    filtered_batches.push(filtered);
                }

                mask_offset += batch_len;
            }
        }

        if filtered_batches.is_empty() {
            // Return empty table with same schema
            let schema = self.schema().ok_or_else(|| {
                crate::error::CylonError::new(crate::error::Code::Invalid,
                    "Table has no schema".to_string())
            })?;

            // Create empty batch
            let empty_columns: Vec<arrow::array::ArrayRef> = schema.fields().iter()
                .map(|field| {
                    arrow::array::new_empty_array(field.data_type())
                })
                .collect();

            let empty_batch = RecordBatch::try_new(schema, empty_columns)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to create empty batch: {}", e),
                ))?;

            Table::from_record_batch(self.ctx.clone(), empty_batch)
        } else {
            Table::from_record_batches(self.ctx.clone(), filtered_batches)
        }
    }

    /// Add a column to the table at a specific position
    /// Corresponds to C++ Table::AddColumn (table.cpp:1613-1624)
    ///
    /// # Arguments
    /// * `position` - Position to insert the column (0-based index)
    /// * `column_name` - Name for the new column
    /// * `input_column` - Array data for the new column
    ///
    /// # Returns
    /// A new Table with the column added
    pub fn add_column(
        &self,
        position: i32,
        column_name: &str,
        input_column: Arc<dyn arrow::array::Array>,
    ) -> CylonResult<Table> {
        use arrow::datatypes::Field;
        use arrow::compute::concat_batches;

        // Check column length matches table rows (C++ table.cpp:1615-1618)
        if input_column.len() != self.rows() as usize {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "New column length must match the number of rows in the table".to_string(),
            ));
        }

        // Get schema
        let schema = self.schema().ok_or_else(|| {
            crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "Table has no schema".to_string(),
            )
        })?;

        // Validate position
        let num_cols = schema.fields().len() as i32;
        if position < 0 || position > num_cols {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                format!("Position {} is out of bounds for table with {} columns", position, num_cols),
            ));
        }

        // Concatenate all batches into one (similar to C++ working with single arrow::Table)
        let combined_batch = if self.batches.len() == 1 {
            self.batches[0].clone()
        } else {
            concat_batches(&schema, &self.batches)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to concatenate batches: {}", e),
                ))?
        };

        // Create new field (C++ table.cpp:1619)
        let new_field = Arc::new(Field::new(column_name, input_column.data_type().clone(), true));

        // Build new schema with field at position
        let mut new_fields: Vec<Arc<Field>> = Vec::with_capacity(num_cols as usize + 1);
        for (i, field) in schema.fields().iter().enumerate() {
            if i == position as usize {
                new_fields.push(new_field.clone());
            }
            new_fields.push(field.clone());
        }
        if position == num_cols {
            new_fields.push(new_field);
        }
        let new_schema = Arc::new(Schema::new(new_fields));

        // Build new columns with input_column at position (C++ table.cpp:1620-1622)
        let mut new_columns: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(num_cols as usize + 1);
        for (i, col) in combined_batch.columns().iter().enumerate() {
            if i == position as usize {
                new_columns.push(input_column.clone());
            }
            new_columns.push(col.clone());
        }
        if position == num_cols {
            new_columns.push(input_column);
        }

        // Create new batch
        let new_batch = RecordBatch::try_new(new_schema, new_columns)
            .map_err(|e| crate::error::CylonError::new(
                crate::error::Code::ExecutionError,
                format!("Failed to create batch with new column: {}", e),
            ))?;

        Table::from_record_batch(self.ctx.clone(), new_batch)
    }

    /// Get the Arrow index
    /// Corresponds to C++ Table::GetArrowIndex (table.cpp:1570)
    pub fn get_arrow_index(&self) -> Option<Arc<dyn BaseArrowIndex>> {
        self.base_arrow_index.clone()
    }

    /// Set the Arrow index for this table
    /// Corresponds to C++ Table::SetArrowIndex (table.cpp:1572-1593)
    ///
    /// # Arguments
    /// * `index` - The index to set
    /// * `drop_index` - If true, remove the index column from the table
    pub fn set_arrow_index(
        &mut self,
        index: Arc<dyn BaseArrowIndex>,
        drop_index: bool,
    ) -> CylonResult<()> {
        use arrow::compute::concat_batches;

        // Combine batches if we have multiple (C++ table.cpp:1573-1578)
        if self.batches.len() > 1 {
            let schema = self.schema().ok_or_else(|| {
                crate::error::CylonError::new(
                    crate::error::Code::Invalid,
                    "Table has no schema".to_string(),
                )
            })?;

            let combined = concat_batches(&schema, &self.batches)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to combine batches: {}", e),
                ))?;

            self.batches = vec![combined];
        }

        // Set the index (C++ table.cpp:1580)
        self.base_arrow_index = Some(index.clone());

        // Drop the index column if requested (C++ table.cpp:1582-1590)
        if drop_index {
            let col_id = index.get_col_id();
            if col_id < 0 || col_id >= self.columns() {
                return Err(crate::error::CylonError::new(
                    crate::error::Code::Invalid,
                    format!("Index column {} is out of bounds", col_id),
                ));
            }

            // Remove the column by projecting all other columns
            let mut keep_cols: Vec<usize> = Vec::new();
            for i in 0..self.columns() {
                if i != col_id {
                    keep_cols.push(i as usize);
                }
            }

            let schema = self.schema().unwrap();
            let mut new_batches = Vec::new();

            for batch in &self.batches {
                let mut new_columns = Vec::new();
                let mut new_fields = Vec::new();

                for &col_idx in &keep_cols {
                    new_columns.push(batch.column(col_idx).clone());
                    new_fields.push(schema.field(col_idx).clone());
                }

                let new_schema = Arc::new(Schema::new(new_fields));
                let new_batch = RecordBatch::try_new(new_schema, new_columns)
                    .map_err(|e| crate::error::CylonError::new(
                        crate::error::Code::ExecutionError,
                        format!("Failed to remove index column: {}", e),
                    ))?;

                new_batches.push(new_batch);
            }

            self.batches = new_batches;
        }

        Ok(())
    }

    /// Reset the Arrow index
    /// Corresponds to C++ Table::ResetArrowIndex (table.cpp:1595-1610)
    ///
    /// # Arguments
    /// * `drop` - If false, add the current index as a column named "index"
    pub fn reset_arrow_index(&mut self, drop: bool) -> CylonResult<()> {
        use crate::indexing::{ArrowRangeIndex, IndexingType};

        if let Some(index) = &self.base_arrow_index {
            // Check if it's already a range index (C++ table.cpp:1597-1598)
            if index.get_indexing_type() == IndexingType::Range {
                // Already a range index, nothing to do
                return Ok(());
            }

            // Get the current index array (C++ table.cpp:1601)
            let index_arr = index.get_index_array()?;

            // Create new range index (C++ table.cpp:1603)
            let range_index = Arc::new(ArrowRangeIndex::new(0, self.rows() as usize, 1));
            self.base_arrow_index = Some(range_index);

            // If not dropping, add the old index as a column (C++ table.cpp:1604-1607)
            if !drop {
                let new_table = self.add_column(0, "index", index_arr)?;
                self.batches = new_table.batches;
            }
        }

        Ok(())
    }

    /// Print the entire table to a string
    /// Corresponds to C++ Table::PrintToOStream(std::ostream&) (table.cpp:1233-1235)
    pub fn print_to_string(&self) -> CylonResult<String> {
        self.print_to_string_range(0, self.columns(), 0, self.rows(), ',', None)
    }

    /// Print a subset of the table to a string
    /// Corresponds to C++ Table::PrintToOStream(int, int, int64_t, int64_t, ...) (table.cpp:1237-1292)
    ///
    /// # Arguments
    /// * `col1` - Starting column index (inclusive)
    /// * `col2` - Ending column index (exclusive)
    /// * `row1` - Starting row index (inclusive)
    /// * `row2` - Ending row index (exclusive)
    /// * `delimiter` - Character to use as delimiter between columns
    /// * `headers` - Optional custom headers. If None, use schema field names
    pub fn print_to_string_range(
        &self,
        col1: i32,
        col2: i32,
        row1: i64,
        row2: i64,
        delimiter: char,
        headers: Option<Vec<String>>,
    ) -> CylonResult<String> {
        use crate::util::to_string::array_to_string;

        let mut output = String::new();

        // Validate column range
        if col1 < 0 || col2 > self.columns() || col1 >= col2 {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                format!("Invalid column range: [{}, {})", col1, col2),
            ));
        }

        // Validate row range
        if row1 < 0 || row2 > self.rows() || row1 > row2 {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                format!("Invalid row range: [{}, {})", row1, row2),
            ));
        }

        let schema = self.schema().ok_or_else(|| {
            crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "Table has no schema".to_string(),
            )
        })?;

        // Print headers (C++ table.cpp:1242-1270)
        if let Some(custom_headers) = &headers {
            // Check if headers match column count
            if custom_headers.len() != self.columns() as usize {
                return Err(crate::error::CylonError::new(
                    crate::error::Code::Invalid,
                    format!(
                        "Provided headers doesn't match with the number of columns. Given {}, Expected {}",
                        custom_headers.len(),
                        self.columns()
                    ),
                ));
            }

            for col in col1..col2 {
                output.push_str(&custom_headers[col as usize]);
                if col != col2 - 1 {
                    output.push(delimiter);
                } else {
                    output.push('\n');
                }
            }
        } else {
            // Use schema field names
            for col in col1..col2 {
                output.push_str(schema.field(col as usize).name());
                if col != col2 - 1 {
                    output.push(delimiter);
                } else {
                    output.push('\n');
                }
            }
        }

        // Print data rows (C++ table.cpp:1271-1289)
        for row in row1..row2 {
            for col in col1..col2 {
                // Find which batch and row within that batch
                let mut current_row = row;
                let mut found = false;

                for batch in &self.batches {
                    let batch_rows = batch.num_rows() as i64;
                    if current_row < batch_rows {
                        // This row is in this batch
                        let array = batch.column(col as usize);
                        let value_str = array_to_string(array.as_ref(), current_row as usize);
                        output.push_str(&value_str);
                        found = true;
                        break;
                    }
                    current_row -= batch_rows;
                }

                if !found {
                    output.push_str("?");
                }

                if col != col2 - 1 {
                    output.push(delimiter);
                }
            }
            output.push('\n');
        }

        Ok(output)
    }

    /// Print the entire table to stdout
    pub fn print(&self) -> CylonResult<()> {
        let output = self.print_to_string()?;
        print!("{}", output);
        Ok(())
    }

    /// Print a subset of the table to stdout
    pub fn print_range(
        &self,
        col1: i32,
        col2: i32,
        row1: i64,
        row2: i64,
    ) -> CylonResult<()> {
        let output = self.print_to_string_range(col1, col2, row1, row2, ',', None)?;
        print!("{}", output);
        Ok(())
    }

    // =========================================================================
    // DataFusion Integration (optional feature)
    // =========================================================================

    /// Convert this Table to a DataFusion DataFrame (zero-copy)
    ///
    /// This enables using DataFusion's DataFrame API for analytics operations
    /// on Cylon tables. The conversion is zero-copy since both Cylon and
    /// DataFusion use arrow-rs RecordBatches internally.
    ///
    /// # Example
    /// ```no_run
    /// use datafusion::prelude::*;
    ///
    /// let table = Table::from_csv_default(ctx, "data.csv")?;
    /// let df = table.to_datafusion().await?;
    ///
    /// // Use DataFusion DataFrame API
    /// let result = df
    ///     .filter(col("age").gt(lit(21)))?
    ///     .select(vec![col("name"), col("age")])?
    ///     .collect()
    ///     .await?;
    /// ```
    #[cfg(feature = "datafusion")]
    pub async fn to_datafusion(&self) -> CylonResult<datafusion::dataframe::DataFrame> {
        use datafusion::prelude::*;
        use datafusion::datasource::MemTable;
        use crate::error::{CylonError, Code};

        let schema = self.schema().ok_or_else(|| {
            CylonError::new(Code::Invalid, "Table has no schema".to_string())
        })?;

        // Create a MemTable from our batches (zero-copy - just Arc references)
        let mem_table = MemTable::try_new(schema, vec![self.batches.clone()])
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to create MemTable: {}", e)))?;

        // Create a SessionContext and register the table
        let session_ctx = SessionContext::new();
        session_ctx.register_table("cylon_table", Arc::new(mem_table))
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to register table: {}", e)))?;

        // Return a DataFrame for the table
        session_ctx.table("cylon_table")
            .await
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to create DataFrame: {}", e)))
    }

    /// Create a Table from a DataFusion DataFrame (zero-copy)
    ///
    /// This allows converting the results of DataFusion queries back into
    /// Cylon Tables for distributed operations. The conversion is zero-copy
    /// since both use arrow-rs RecordBatches.
    ///
    /// # Example
    /// ```no_run
    /// use datafusion::prelude::*;
    ///
    /// let session_ctx = SessionContext::new();
    /// let df = session_ctx.read_csv("data.csv", CsvReadOptions::new()).await?;
    /// let filtered = df.filter(col("value").gt(lit(100)))?;
    ///
    /// // Convert back to Cylon Table for distributed operations
    /// let table = Table::from_datafusion(cylon_ctx, filtered).await?;
    /// let shuffled = cylon::table::shuffle(&table, &[0])?;
    /// ```
    #[cfg(feature = "datafusion")]
    pub async fn from_datafusion(
        ctx: Arc<CylonContext>,
        df: datafusion::dataframe::DataFrame,
    ) -> CylonResult<Self> {
        use crate::error::{CylonError, Code};

        // Collect the DataFrame into RecordBatches (zero-copy for in-memory data)
        let batches = df.collect()
            .await
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to collect DataFrame: {}", e)))?;

        Table::from_record_batches(ctx, batches)
    }

    // =========================================================================
    // Polars Integration (optional feature)
    // =========================================================================

    /// Convert this Table to a Polars DataFrame (zero-copy via Arrow C Data Interface)
    ///
    /// This enables using Polars DataFrame API for analytics operations on Cylon tables.
    /// The conversion uses the Arrow C Data Interface for zero-copy data transfer between
    /// `arrow-rs` (used by Cylon) and `polars-arrow` (used by Polars).
    ///
    /// # Example
    /// ```no_run
    /// use polars::prelude::*;
    ///
    /// let table = Table::from_csv_default(ctx, "data.csv")?;
    /// let df = table.to_polars()?;
    ///
    /// // Use Polars DataFrame API
    /// let result = df.filter(&df.column("age")?.gt(21)?)?;
    /// ```
    #[cfg(feature = "polars")]
    pub fn to_polars(&self) -> CylonResult<polars::frame::DataFrame> {
        use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
        use polars::prelude::*;
        use polars_arrow::ffi;
        use crate::error::{CylonError, Code};

        let schema = self.schema().ok_or_else(|| {
            CylonError::new(Code::Invalid, "Table has no schema".to_string())
        })?;

        // Collect all columns across all batches
        let mut all_columns: Vec<Column> = Vec::with_capacity(schema.fields().len());

        for (col_idx, field) in schema.fields().iter().enumerate() {
            // Collect all arrays for this column across batches
            let mut polars_chunks: Vec<Box<dyn polars_arrow::array::Array>> = Vec::new();

            for batch in &self.batches {
                let column = batch.column(col_idx);

                // Export arrow-rs array to C Data Interface
                let data = column.to_data();
                let mut ffi_array = FFI_ArrowArray::new(&data);
                let mut ffi_schema = FFI_ArrowSchema::try_from(field.data_type())
                    .map_err(|e| CylonError::new(Code::ExecutionError,
                        format!("Failed to export schema to FFI: {}", e)))?;

                // Import into polars-arrow via C Data Interface
                // SAFETY: We're using the stable Arrow C Data Interface ABI
                // The FFI structs are ABI-compatible between arrow-rs and polars-arrow
                unsafe {
                    // Transmute arrow-rs FFI structs to polars-arrow FFI structs
                    // They are ABI-compatible #[repr(C)] structs per Arrow C Data Interface
                    let polars_ffi_schema: polars_arrow::ffi::ArrowSchema =
                        std::mem::transmute(ffi_schema);
                    let polars_ffi_array: polars_arrow::ffi::ArrowArray =
                        std::mem::transmute(ffi_array);

                    let polars_field = ffi::import_field_from_c(&polars_ffi_schema)
                        .map_err(|e| CylonError::new(Code::ExecutionError,
                            format!("Failed to import field from FFI: {}", e)))?;

                    let polars_array = ffi::import_array_from_c(
                        polars_ffi_array,
                        polars_field.dtype().clone()
                    ).map_err(|e| CylonError::new(Code::ExecutionError,
                        format!("Failed to import array from FFI: {}", e)))?;

                    polars_chunks.push(polars_array);
                }
            }

            // Create a Series from the chunks, then convert to Column
            let series = Series::try_from((PlSmallStr::from(field.name().as_str()), polars_chunks))
                .map_err(|e| CylonError::new(Code::ExecutionError,
                    format!("Failed to create Series '{}': {}", field.name(), e)))?;

            all_columns.push(Column::from(series));
        }

        polars::frame::DataFrame::new(all_columns)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to create DataFrame: {}", e)))
    }

    /// Create a Table from a Polars DataFrame (zero-copy via Arrow C Data Interface)
    ///
    /// This allows converting Polars DataFrame results back into Cylon Tables
    /// for distributed operations. The conversion uses the Arrow C Data Interface
    /// for zero-copy data transfer.
    ///
    /// # Example
    /// ```no_run
    /// use polars::prelude::*;
    ///
    /// let df = CsvReadOptions::default()
    ///     .try_into_reader_with_file_path(Some("data.csv".into()))?
    ///     .finish()?;
    ///
    /// // Convert to Cylon Table for distributed operations
    /// let table = Table::from_polars(cylon_ctx, &df)?;
    /// ```
    #[cfg(feature = "polars")]
    pub fn from_polars(ctx: Arc<CylonContext>, df: &polars::frame::DataFrame) -> CylonResult<Self> {
        use arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
        use arrow::datatypes::{Field, Schema};
        use polars::prelude::*;
        use polars_arrow::ffi;
        use crate::error::{CylonError, Code};

        let mut arrow_fields: Vec<Field> = Vec::with_capacity(df.width());
        let mut arrow_columns: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(df.width());

        for column in df.get_columns() {
            // Get the materialized Series from the Column
            let series = column.as_materialized_series();
            let name = series.name().to_string();
            let n_chunks = series.n_chunks();

            // Get chunks from the series using array_ref
            for chunk_idx in 0..n_chunks {
                let chunk = series.array_ref(chunk_idx);

                // Export polars-arrow array to C Data Interface
                let polars_ffi_array = ffi::export_array_to_c(chunk.clone());

                // We also need the field/schema - get it from the series dtype
                let polars_field = polars_arrow::datatypes::Field::new(
                    PlSmallStr::from(name.as_str()),
                    series.dtype().to_arrow(CompatLevel::newest()),
                    true, // is_nullable
                );
                let polars_ffi_schema = ffi::export_field_to_c(&polars_field);

                // Import into arrow-rs via C Data Interface
                // SAFETY: We're using the stable Arrow C Data Interface ABI
                unsafe {
                    // Convert the polars FFI types to arrow-rs FFI types
                    // They are ABI-compatible #[repr(C)] structs
                    let arrow_ffi_array: FFI_ArrowArray = std::mem::transmute(polars_ffi_array);
                    let arrow_ffi_schema: FFI_ArrowSchema = std::mem::transmute(polars_ffi_schema);

                    let arrow_data = from_ffi(arrow_ffi_array, &arrow_ffi_schema)
                        .map_err(|e| CylonError::new(Code::ExecutionError,
                            format!("Failed to import array from FFI: {}", e)))?;

                    let arrow_array = arrow::array::make_array(arrow_data);

                    // Extract the field from the schema for first chunk only
                    if chunk_idx == 0 {
                        let field = Field::new(&name, arrow_array.data_type().clone(), true);
                        arrow_fields.push(field);
                    }

                    arrow_columns.push(arrow_array);
                }
            }
        }

        // Build schema and RecordBatch
        let schema = Arc::new(Schema::new(arrow_fields));

        // For simplicity, create one batch (Polars may have multiple chunks but
        // we consolidate them here)
        let batch = RecordBatch::try_new(schema, arrow_columns)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to create RecordBatch: {}", e)))?;

        Table::from_record_batch(ctx, batch)
    }

}

// Set operations - standalone functions (not Table methods)
// Ported from cpp/src/cylon/table.cpp lines 925-1107

/// Union of two tables (unique rows from both tables)
/// Corresponds to C++ Union function (table.cpp:925-995)
///
/// Returns unique rows from both tables combined
///
/// # Arguments
/// * `first` - First table
/// * `second` - Second table
///
/// # Returns
/// A new table containing unique rows from both tables
pub fn union(first: &Table, second: &Table) -> CylonResult<Table> {
    use hashbrown::HashSet;
    use arrow::compute::concat_batches;
    use crate::arrow::arrow_comparator::{DualTableRowIndexHash, DualTableRowIndexEqualTo, set_bit};
    use arrow::array::BooleanBuilder;
    use crate::error::{CylonError, Code};

    // Combine batches for both tables (C++ lines 932-933)
    let schema = first.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "First table has no schema".to_string())
    })?;

    let left_batch = if first.batches.len() > 1 {
        concat_batches(&schema, &first.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine left batches: {}", e)))?
    } else if first.batches.len() == 1 {
        first.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "First table has no batches".to_string()));
    };

    let other_schema = second.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Second table has no schema".to_string())
    })?;

    // Verify schemas match (C++ line 935)
    if !schema.eq(&other_schema) {
        return Err(CylonError::new(Code::Invalid,
            "Tables must have the same schema for union".to_string()));
    }

    let right_batch = if second.batches.len() > 1 {
        concat_batches(&other_schema, &second.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine right batches: {}", e)))?
    } else if second.batches.len() == 1 {
        second.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "Second table has no batches".to_string()));
    };

    // Create hash and equality infrastructure (C++ lines 937-941)
    let hash = DualTableRowIndexHash::new(&left_batch, &right_batch)?;
    let equal_to = DualTableRowIndexEqualTo::new(&left_batch, &right_batch)?;

    // Create hash set with custom hash and equality (C++ lines 943-945)
    let mut rows_set = HashSet::with_capacity_and_hasher(
        (left_batch.num_rows() + right_batch.num_rows()) as usize,
        std::hash::BuildHasherDefault::<ahash::AHasher>::default(),
    );

    // Insert left table rows and track unique ones (C++ lines 952-960)
    let mut left_mask = BooleanBuilder::with_capacity(left_batch.num_rows());
    for i in 0..left_batch.num_rows() as i64 {
        let hash_val = hash.hash(i);
        // Check if this row is unique by attempting to insert
        let is_unique = if let Some(&existing_idx) = rows_set.iter().find(|&&idx| {
            hash.hash(idx) == hash_val && equal_to.equal(idx, i)
        }) {
            false // Already exists
        } else {
            rows_set.insert(i);
            true
        };
        left_mask.append_value(is_unique); // C++ line 958
    }
    let left_mask_array = left_mask.finish();

    // Filter left table (C++ lines 962-966)
    let filtered_left = arrow::compute::filter_record_batch(&left_batch, &left_mask_array)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to filter left table: {}", e)))?;

    // Insert right table rows with bit set and track unique ones (C++ lines 969-979)
    let mut right_mask = BooleanBuilder::with_capacity(right_batch.num_rows());
    for i in 0..right_batch.num_rows() as i64 {
        let idx = set_bit(i); // C++ line 974
        let hash_val = hash.hash(idx);
        // Check if this row is unique
        let is_unique = if let Some(&existing_idx) = rows_set.iter().find(|&&existing| {
            hash.hash(existing) == hash_val && equal_to.equal(existing, idx)
        }) {
            false // Already exists
        } else {
            rows_set.insert(idx);
            true
        };
        right_mask.append_value(is_unique); // C++ line 977
    }
    let right_mask_array = right_mask.finish();

    // Filter right table (C++ lines 981-984)
    let filtered_right = arrow::compute::filter_record_batch(&right_batch, &right_mask_array)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to filter right table: {}", e)))?;

    // Concatenate filtered tables (C++ lines 987-990)
    let result_batches = vec![filtered_left, filtered_right];
    let concatenated = concat_batches(&schema, &result_batches)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to concatenate tables: {}", e)))?;

    Table::from_record_batch(first.ctx.clone(), concatenated)
}

/// Subtract - rows in first table that are not in second table
/// Corresponds to C++ Subtract function (table.cpp:997-1049)
///
/// Returns rows from first table that don't exist in second table
///
/// # Arguments
/// * `first` - First table
/// * `second` - Second table (rows to subtract)
///
/// # Returns
/// A new table containing rows from first that are not in second
pub fn subtract(first: &Table, second: &Table) -> CylonResult<Table> {
    use hashbrown::HashSet;
    use arrow::compute::concat_batches;
    use crate::arrow::arrow_comparator::{DualTableRowIndexHash, DualTableRowIndexEqualTo, set_bit};
    use arrow::array::BooleanBuilder;
    use crate::error::{CylonError, Code};

    // Combine batches for both tables (C++ lines 1004-1005)
    let schema = first.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "First table has no schema".to_string())
    })?;

    let left_batch = if first.batches.len() > 1 {
        concat_batches(&schema, &first.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine left batches: {}", e)))?
    } else if first.batches.len() == 1 {
        first.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "First table has no batches".to_string()));
    };

    let other_schema = second.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Second table has no schema".to_string())
    })?;

    // Verify schemas match (C++ line 1007)
    if !schema.eq(&other_schema) {
        return Err(CylonError::new(Code::Invalid,
            "Tables must have the same schema for subtract".to_string()));
    }

    let right_batch = if second.batches.len() > 1 {
        concat_batches(&other_schema, &second.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine right batches: {}", e)))?
    } else if second.batches.len() == 1 {
        second.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "Second table has no batches".to_string()));
    };

    // Create hash and equality infrastructure (C++ lines 1009-1013)
    let hash = DualTableRowIndexHash::new(&left_batch, &right_batch)?;
    let equal_to = DualTableRowIndexEqualTo::new(&left_batch, &right_batch)?;

    // Create hash set (C++ lines 1015-1017)
    let mut rows_set = HashSet::with_capacity_and_hasher(
        left_batch.num_rows() as usize,
        std::hash::BuildHasherDefault::<ahash::AHasher>::default(),
    );

    // Insert left table rows and create initial mask (C++ lines 1026-1029)
    let mut mask_builder = BooleanBuilder::with_capacity(left_batch.num_rows());
    for i in 0..left_batch.num_rows() as i64 {
        let hash_val = hash.hash(i);
        // Check if this row is unique
        let is_unique = if let Some(&_existing_idx) = rows_set.iter().find(|&&idx| {
            hash.hash(idx) == hash_val && equal_to.equal(idx, i)
        }) {
            false // Already exists
        } else {
            rows_set.insert(i);
            true
        };
        mask_builder.append_value(is_unique); // C++ line 1028
    }
    let mask_array = mask_builder.finish();

    // Get the mask as a vector so we can mutate it (C++ line 1033)
    let mut bitmask: Vec<bool> = (0..mask_array.len())
        .map(|i| mask_array.value(i))
        .collect();

    // Probe right rows and clear bits for matches (C++ lines 1036-1042)
    for i in 0..right_batch.num_rows() as i64 {
        let idx = set_bit(i); // C++ line 1038
        let hash_val = hash.hash(idx);

        // Find matching row in left table
        if let Some(&left_idx) = rows_set.iter().find(|&&left| {
            hash.hash(left) == hash_val && equal_to.equal(left, idx)
        }) {
            // Clear the bit for this row (C++ line 1040)
            bitmask[left_idx as usize] = false;
        }
    }

    // Reconstruct the boolean array
    let mut mask_builder = BooleanBuilder::with_capacity(bitmask.len());
    for &bit in &bitmask {
        mask_builder.append_value(bit);
    }
    let mask_array = mask_builder.finish();

    // Filter left table (C++ lines 1045-1048)
    let filtered = arrow::compute::filter_record_batch(&left_batch, &mask_array)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to filter table: {}", e)))?;

    Table::from_record_batch(first.ctx.clone(), filtered)
}

/// Intersect - rows that exist in both tables
/// Corresponds to C++ Intersect function (table.cpp:1051-1110)
///
/// Returns rows that exist in both tables
///
/// # Arguments
/// * `first` - First table
/// * `second` - Second table
///
/// # Returns
/// A new table containing rows that exist in both tables
pub fn intersect(first: &Table, second: &Table) -> CylonResult<Table> {
    use hashbrown::HashSet;
    use arrow::compute::concat_batches;
    use crate::arrow::arrow_comparator::{DualTableRowIndexHash, DualTableRowIndexEqualTo, set_bit};
    use arrow::array::BooleanBuilder;
    use crate::error::{CylonError, Code};

    // Combine batches for both tables (C++ lines 1060-1061)
    let schema = first.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "First table has no schema".to_string())
    })?;

    let left_batch = if first.batches.len() > 1 {
        concat_batches(&schema, &first.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine left batches: {}", e)))?
    } else if first.batches.len() == 1 {
        first.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "First table has no batches".to_string()));
    };

    let other_schema = second.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Second table has no schema".to_string())
    })?;

    // Verify schemas match (C++ line 1063)
    if !schema.eq(&other_schema) {
        return Err(CylonError::new(Code::Invalid,
            "Tables must have the same schema for intersect".to_string()));
    }

    let right_batch = if second.batches.len() > 1 {
        concat_batches(&other_schema, &second.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine right batches: {}", e)))?
    } else if second.batches.len() == 1 {
        second.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "Second table has no batches".to_string()));
    };

    // Create hash and equality infrastructure (C++ lines 1065-1069)
    let hash = DualTableRowIndexHash::new(&left_batch, &right_batch)?;
    let equal_to = DualTableRowIndexEqualTo::new(&left_batch, &right_batch)?;

    // Create hash set (C++ lines 1071-1073)
    let mut rows_set = HashSet::with_capacity_and_hasher(
        left_batch.num_rows() as usize,
        std::hash::BuildHasherDefault::<ahash::AHasher>::default(),
    );

    // Insert all left table rows into set (C++ lines 1078-1080)
    for i in 0..left_batch.num_rows() as i64 {
        rows_set.insert(i);
    }

    // Create bitmask (all false initially) (C++ line 1083)
    let mut bitmask = vec![false; left_batch.num_rows()];

    // Probe right rows and set bits for matches (C++ lines 1086-1092)
    for i in 0..right_batch.num_rows() as i64 {
        let idx = set_bit(i); // C++ line 1088
        let hash_val = hash.hash(idx);

        // Find matching row in left table
        if let Some(&left_idx) = rows_set.iter().find(|&&left| {
            hash.hash(left) == hash_val && equal_to.equal(left, idx)
        }) {
            // Mark this left row as matching (C++ line 1090)
            bitmask[left_idx as usize] = true;
        }
    }

    // Convert bitmask to BooleanArray (C++ lines 1095-1098)
    let mut mask_builder = BooleanBuilder::with_capacity(bitmask.len());
    for &bit in &bitmask {
        mask_builder.append_value(bit);
    }
    let mask_array = mask_builder.finish();

    // Filter left table (C++ lines 1100-1107)
    let filtered = arrow::compute::filter_record_batch(&left_batch, &mask_array)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to filter table: {}", e)))?;

    Table::from_record_batch(first.ctx.clone(), filtered)
}

/// Unique - remove duplicate rows from a table
/// Corresponds to C++ Unique function (table.cpp:1306-1374)
///
/// Returns a table with only unique rows based on specified columns
///
/// # Arguments
/// * `table` - Input table
/// * `col_indices` - Column indices to use for uniqueness check
/// * `keep_first` - If true, keep first occurrence; if false, keep last occurrence
///
/// # Returns
/// A new table containing only unique rows
pub fn unique(table: &Table, col_indices: &[usize], keep_first: bool) -> CylonResult<Table> {
    use hashbrown::HashSet;
    use arrow::compute::concat_batches;
    use crate::arrow::arrow_comparator::{TableRowIndexHash, TableRowIndexEqualTo};
    use crate::error::{CylonError, Code};

    // Return empty table if input is empty (C++ lines 1370-1372)
    if table.is_empty() {
        return Table::from_record_batches(table.ctx.clone(), table.batches.clone());
    }

    // Combine batches (C++ lines 1316-1318)
    let schema = table.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Table has no schema".to_string())
    })?;

    let batch = if table.batches.len() > 1 {
        concat_batches(&schema, &table.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine batches: {}", e)))?
    } else if table.batches.len() == 1 {
        table.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "Table has no batches".to_string()));
    };

    // Create hash and equality infrastructure (C++ lines 1320-1324)
    let row_hash = TableRowIndexHash::new_with_columns(&batch, col_indices)?;
    let row_comp = TableRowIndexEqualTo::new_with_columns(&batch, col_indices)?;

    let num_rows = batch.num_rows();

    // Create hash set (C++ lines 1327-1328)
    let mut rows_set = HashSet::with_capacity_and_hasher(
        num_rows,
        std::hash::BuildHasherDefault::<ahash::AHasher>::default(),
    );

    // Collect unique row indices (C++ lines 1330-1349)
    let mut unique_indices = Vec::with_capacity(num_rows);

    if keep_first {
        // Keep first occurrence (C++ lines 1336-1341)
        for row in 0..num_rows as i64 {
            let hash_val = row_hash.hash(row);
            // Check if this row is unique
            let is_unique = if let Some(&_existing_idx) = rows_set.iter().find(|&&idx| {
                row_hash.hash(idx) == hash_val && row_comp.equal(idx, row)
            }) {
                false // Already exists
            } else {
                rows_set.insert(row);
                true
            };

            if is_unique {
                unique_indices.push(row); // C++ line 1339
            }
        }
    } else {
        // Keep last occurrence (C++ lines 1343-1348)
        for row in (0..num_rows as i64).rev() {
            let hash_val = row_hash.hash(row);
            // Check if this row is unique
            let is_unique = if let Some(&_existing_idx) = rows_set.iter().find(|&&idx| {
                row_hash.hash(idx) == hash_val && row_comp.equal(idx, row)
            }) {
                false // Already exists
            } else {
                rows_set.insert(row);
                true
            };

            if is_unique {
                unique_indices.push(row); // C++ line 1346
            }
        }
        // Reverse to maintain original order (since we iterated backwards)
        unique_indices.reverse();
    }

    // Take rows at the unique indices (C++ lines 1357-1359)
    use arrow::array::Int64Array;
    use arrow::compute::take;

    let indices_array = Int64Array::from(unique_indices);
    let mut taken_columns = Vec::new();

    for col_idx in 0..batch.num_columns() {
        let taken_col = take(batch.column(col_idx), &indices_array, None)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to take column {}: {}", col_idx, e)))?;
        taken_columns.push(taken_col);
    }

    let result_batch = RecordBatch::try_new(batch.schema(), taken_columns)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to create result batch: {}", e)))?;

    Table::from_record_batch(table.ctx.clone(), result_batch)
}

/// Join two tables
/// Corresponds to C++ Join function (table.cpp:819-854)
///
/// Joins two tables based on the provided configuration
///
/// # Arguments
/// * `left` - Left table
/// * `right` - Right table
/// * `config` - Join configuration (join type, columns, algorithm)
///
/// # Returns
/// A new table containing the joined result
pub fn join(
    left: &Table,
    right: &Table,
    config: &crate::join::JoinConfig,
) -> CylonResult<Table> {
    crate::join::join(left, right, config)
}

/// Sort table by a single column
/// Corresponds to C++ Sort function (table.cpp:372-387)
///
/// Performs local sort on the table. If table has chunked columns, they will be
/// merged in the output.
///
/// # Arguments
/// * `table` - Input table to sort
/// * `sort_column` - Column index to sort by
/// * `ascending` - Sort direction (true = ascending, false = descending)
///
/// # Returns
/// A new sorted table
///
/// # Example
/// ```ignore
/// // Sort by column 0 in ascending order
/// let sorted = sort(&table, 0, true)?;
/// ```
pub fn sort(
    table: &Table,
    sort_column: usize,
    ascending: bool,
) -> CylonResult<Table> {
    table.sort(sort_column, ascending)
}

/// Sort table by multiple columns with same direction
/// Corresponds to C++ Sort function (table.cpp:389-393)
///
/// # Arguments
/// * `table` - Input table to sort
/// * `sort_columns` - Column indices to sort by (in order of priority)
/// * `ascending` - Sort direction for all columns (true = ascending, false = descending)
///
/// # Returns
/// A new sorted table
pub fn sort_multi_same_direction(
    table: &Table,
    sort_columns: &[usize],
    ascending: bool,
) -> CylonResult<Table> {
    let sort_directions = vec![ascending; sort_columns.len()];
    table.sort_multi(sort_columns, &sort_directions)
}

/// Sort table by multiple columns with individual directions
/// Corresponds to C++ Sort function (table.cpp:395-418)
///
/// # Arguments
/// * `table` - Input table to sort
/// * `sort_columns` - Column indices to sort by (in order of priority)
/// * `sort_directions` - Sort direction for each column (true = ascending, false = descending)
///
/// # Returns
/// A new sorted table
///
/// # Example
/// ```ignore
/// // Sort by column 0 ascending, then column 1 descending
/// let sorted = sort_multi(&table, &[0, 1], &[true, false])?;
/// ```
pub fn sort_multi(
    table: &Table,
    sort_columns: &[usize],
    sort_directions: &[bool],
) -> CylonResult<Table> {
    table.sort_multi(sort_columns, sort_directions)
}

/// Project (select) specific columns from a table
/// Corresponds to C++ Project function (table.cpp:1212-1231)
///
/// Creates a new table containing only the specified columns.
/// This is also known as column selection or vertical slicing.
///
/// # Arguments
/// * `table` - Input table
/// * `project_columns` - Indices of columns to include in the result
///
/// # Returns
/// A new table with only the specified columns
///
/// # Example
/// ```ignore
/// // Select columns 0, 2, and 3 from the table
/// let projected = project(&table, &[0, 2, 3])?;
/// ```
pub fn project(
    table: &Table,
    project_columns: &[usize],
) -> CylonResult<Table> {
    table.project(project_columns)
}

/// Slice a table to extract a range of rows
/// Corresponds to C++ Slice function (slice.cpp:33-45)
///
/// Extracts rows from offset to offset+length.
///
/// # Arguments
/// * `table` - Input table
/// * `offset` - Starting row index (0-based)
/// * `length` - Number of rows to include
///
/// # Returns
/// A new table with the specified row range
///
/// # Example
/// ```ignore
/// // Get rows 10-19 (10 rows starting at index 10)
/// let sliced = slice(&table, 10, 10)?;
/// ```
pub fn slice(
    table: &Table,
    offset: usize,
    length: usize,
) -> CylonResult<Table> {
    table.slice(offset, length)
}

/// Get the first n rows of a table
/// Corresponds to C++ Head function (slice.cpp:106-112)
///
/// Returns a new table with the first n rows.
///
/// # Arguments
/// * `table` - Input table
/// * `num_rows` - Number of rows to return from the beginning
///
/// # Returns
/// A new table with the first num_rows rows
///
/// # Example
/// ```ignore
/// // Get first 5 rows
/// let top5 = head(&table, 5)?;
/// ```
pub fn head(
    table: &Table,
    num_rows: usize,
) -> CylonResult<Table> {
    table.head(num_rows)
}

/// Get the last n rows of a table
/// Corresponds to C++ Tail function (slice.cpp:131-142)
///
/// Returns a new table with the last n rows.
///
/// # Arguments
/// * `table` - Input table
/// * `num_rows` - Number of rows to return from the end
///
/// # Returns
/// A new table with the last num_rows rows
///
/// # Example
/// ```ignore
/// // Get last 10 rows
/// let bottom10 = tail(&table, 10)?;
/// ```
pub fn tail(
    table: &Table,
    num_rows: usize,
) -> CylonResult<Table> {
    table.tail(num_rows)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortAlgorithm {
    Sample,
    Initial,
}

#[derive(Clone, Copy, Debug)]
pub struct SortOptions {
    pub algorithm: SortAlgorithm,
    pub num_samples: i64,
    pub num_bins: i32,
}

impl Default for SortOptions {
    fn default() -> Self {
        SortOptions {
            algorithm: SortAlgorithm::Sample,
            num_samples: 0, // default in C++ is 2, but let's use 0 to mean default
            num_bins: 256,
        }
    }
}

/// Distributed sort of a table
/// Corresponds to C++ DistributedSort function (table.cpp:1165)
///
/// # Arguments
/// * `table` - Input table to sort
/// * `sort_columns` - Column indices to sort by
/// * `sort_directions` - Sort direction for each column (true = ascending)
/// * `sort_options` - Additional options for sorting algorithm
///
/// # Returns
/// A new globally sorted table, distributed across all processes
pub fn distributed_sort(
    table: &Table,
    sort_columns: &[usize],
    sort_directions: &[bool],
    sort_options: SortOptions,
) -> CylonResult<Table> {
    if sort_columns.len() != sort_directions.len() {
        return Err(crate::error::CylonError::new(
            crate::error::Code::Invalid,
            "sort_columns and sort_directions must have the same length".to_string(),
        ));
    }

    let ctx = table.get_context();
    let world_size = ctx.get_world_size() as usize;

    if world_size == 1 {
        return table.sort_multi(sort_columns, sort_directions);
    }

    match sort_options.algorithm {
        SortAlgorithm::Sample => {
            // Implementation of DistributedSortRegularSampling (table.cpp:1045)

            // 1. Local sort (C++: 1061-1063)
            let local_sorted = table.sort_multi(sort_columns, sort_directions)?;

            // 2. Sample the sorted table (C++: 1065-1074)
            let num_samples = if sort_options.num_samples == 0 {
                (world_size * 2) as i64 // just a default, C++ uses a SAMPLING_RATIO
            } else {
                sort_options.num_samples
            };
            let sample_count = std::cmp::min(num_samples, local_sorted.rows());

            let sample_result =
                crate::util::arrow_utils::sample_table_uniform(&local_sorted, sample_count as usize, Some(sort_columns))?;

            // 3. Determine split points (C++: 1075-1082)
            let split_points = get_split_points(&sample_result, sort_directions)?;

            // 4. Partition local data based on split points (C++: 1085-1088)
            let (target_partitions, partition_hist) =
                get_split_point_indices(&split_points, &local_sorted, sort_columns, sort_directions)?;

            // 5. All-to-all exchange (C++: 1091-1106)
            let split_batches =
                split_by_partitions(&local_sorted, world_size, &target_partitions, &partition_hist)?;

            let mut serialized_partitions = Vec::with_capacity(world_size);
            for batch in split_batches {
                serialized_partitions.push(serialize_record_batch(&batch)?);
            }

            let comm = ctx.get_communicator().ok_or_else(|| {
                crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    "Communicator not initialized".to_string(),
                )
            })?;
            let received_serialized = comm.all_to_all(serialized_partitions)?;

            let mut received_batches = Vec::new();
            for serialized_batch in received_serialized {
                if !serialized_batch.is_empty() {
                    received_batches.push(deserialize_record_batch(&serialized_batch)?);
                }
            }

            // 6. Final local merge sort (C++: 1108)
            let mut received_tables = Vec::new();
            for batch in received_batches {
                if batch.num_rows() > 0 {
                    received_tables.push(Table::from_record_batch(ctx.clone(), batch)?);
                }
            }

            let received_tables_ref: Vec<&Table> = received_tables.iter().collect();
            if received_tables_ref.is_empty() {
                // if all received tables are empty, create an empty table with the correct schema
                let schema = table.schema().ok_or_else(|| {
                    crate::error::CylonError::new(
                        crate::error::Code::Invalid,
                        "Table has no schema".to_string(),
                    )
                })?;
                let empty_batch = arrow::record_batch::RecordBatch::new_empty(schema);
                return Table::from_record_batch(ctx.clone(), empty_batch);
            }

            let final_sorted =
                merge_sorted_table(&received_tables_ref, sort_columns, sort_directions)?;

            Ok(final_sorted)
        }
        SortAlgorithm::Initial => {
            // Implementation of DistributedSortInitialSampling (table.cpp:1112)
            unimplemented!("Distributed initial sampling sort not yet implemented");
        }
    }
}

use crate::net::serialize::{deserialize_table, serialize_table};

/// Determines split points for distributed sort.
/// It gathers samples from all workers, merges them, samples again, and broadcasts the final splitters.
fn get_split_points(sample_result: &Table, sort_directions: &[bool]) -> CylonResult<Table> {
    let ctx = sample_result.get_context();
    let comm = ctx.get_communicator().ok_or_else(|| {
        crate::error::CylonError::new(
            crate::error::Code::ExecutionError,
            "Communicator not initialized".to_string(),
        )
    })?;

    // 1. Serialize local samples
    let serialized_samples = serialize_table(sample_result)?;

    // 2. Allgather serialized samples
    let all_serialized_samples = comm.allgather(&serialized_samples)?;

    let rank = ctx.get_rank();
    let mut split_points: Option<Table> = None;

    if rank == 0 {
        // 3. On root, deserialize, merge, and sample to get split points
        let mut gathered_tables = Vec::new();
        for serialized_table in all_serialized_samples {
            let table = deserialize_table(ctx.clone(), &serialized_table)?;
            gathered_tables.push(table);
        }

        let gathered_tables_ref: Vec<&Table> = gathered_tables.iter().collect();

        // Merge sorted samples
        let sort_columns: Vec<usize> = (0..sample_result.columns() as usize).collect();
        let merged_samples = merge_sorted_table(&gathered_tables_ref, &sort_columns, sort_directions)?;

        // Sample again to get final splitters
        let num_split_points = std::cmp::min(merged_samples.rows(), (ctx.get_world_size() - 1) as i64);
        let final_splitters =
            crate::util::arrow_utils::sample_table_uniform(&merged_samples, num_split_points as usize, Some(&sort_columns))?;
        split_points = Some(final_splitters);
    }

    // 4. Broadcast split points from root to all
    comm.bcast(&mut split_points, 0, ctx.clone())?;

    split_points.ok_or_else(|| {
        crate::error::CylonError::new(
            crate::error::Code::ExecutionError,
            "Failed to receive broadcasted split points".to_string(),
        )
    })
}

/// Placeholder for calculating partition indices based on split points.
/// This function should perform a binary search for each split point on the sorted table
/// to determine the boundaries for partitioning.
fn get_split_point_indices(
    _split_points: &Table,
    sorted_table: &Table,
    _sort_columns: &[usize],
    _sort_directions: &[bool],
) -> CylonResult<(Vec<u32>, Vec<u32>)> {
    // This is a placeholder. A real implementation would do a binary search.
    // For now, we just split the table into world_size equal partitions.
    let world_size = sorted_table.get_context().get_world_size() as usize;
    let num_rows = sorted_table.rows() as usize;
    let mut target_partitions = vec![0u32; num_rows];
    let mut partition_hist = vec![0u32; world_size];

    if num_rows == 0 {
        return Ok((target_partitions, partition_hist));
    }

    let rows_per_partition = num_rows / world_size;
    let remainder = num_rows % world_size;

    let mut current_pos = 0;
    for i in 0..world_size {
        let mut partition_size = rows_per_partition;
        if i < remainder {
            partition_size += 1;
        }
        for j in 0..partition_size {
            if current_pos + j < num_rows {
                target_partitions[current_pos + j] = i as u32;
            }
        }
        partition_hist[i] = partition_size as u32;
        current_pos += partition_size;
    }

    Ok((target_partitions, partition_hist))
}

fn split_by_partitions(
    table: &Table,
    num_partitions: usize,
    target_partitions: &[u32],
    partition_hist: &[u32],
) -> CylonResult<Vec<arrow::record_batch::RecordBatch>> {
    use arrow::compute::concat_batches;

    let combined_batch = if table.num_batches() > 1 {
        concat_batches(&table.schema().unwrap(), table.batches())?
    } else {
        table.batch(0).unwrap().clone()
    };

    let schema = combined_batch.schema();

    // Build indices for each partition
    let mut partition_indices: Vec<Vec<usize>> = vec![Vec::new(); num_partitions];
    for part_idx in 0..num_partitions {
        partition_indices[part_idx].reserve(partition_hist[part_idx] as usize);
    }

    for (row_idx, &partition) in target_partitions.iter().enumerate() {
        partition_indices[partition as usize].push(row_idx);
    }

    let mut result_batches = Vec::new();

    for partition_idx in 0..num_partitions {
        let indices = &partition_indices[partition_idx];

        if indices.is_empty() {
            result_batches.push(arrow::record_batch::RecordBatch::new_empty(schema.clone()));
        } else {
            let indices_array = arrow::array::UInt64Array::from(
                indices.iter().map(|&i| i as u64).collect::<Vec<_>>(),
            );

            let mut partition_columns = Vec::new();
            for col_idx in 0..combined_batch.num_columns() {
                let column = combined_batch.column(col_idx);
                let taken = arrow::compute::take(column.as_ref(), &indices_array, None)?;
                partition_columns.push(taken);
            }

            let partition_batch = arrow::record_batch::RecordBatch::try_new(
                schema.clone(),
                partition_columns,
            )?;
            result_batches.push(partition_batch);
        }
    }

    Ok(result_batches)
}

/// Merge multiple tables vertically (concatenate rows)
/// Corresponds to C++ Merge function (table.cpp:343-370)
///
/// Concatenates multiple tables vertically into a single table.
/// All tables must have the same schema.
///
/// # Arguments
/// * `tables` - Vector of tables to merge
///
/// # Returns
/// A new table containing all rows from all input tables
///
/// # Example
/// ```ignore
/// // Merge three tables into one
/// let merged = merge(&[&table1, &table2, &table3])?;
/// ```
pub fn merge(tables: &[&Table]) -> CylonResult<Table> {
    if tables.is_empty() {
        return Err(crate::error::CylonError::new(
            crate::error::Code::Invalid,
            "Cannot merge empty vector of tables".to_string()
        ));
    }

    // Filter out empty tables (matches C++ behavior at table.cpp:349-350)
    let non_empty: Vec<&Table> = tables.iter()
        .filter(|t| t.rows() > 0)
        .copied()
        .collect();

    // If all tables are empty, return the first table (matches C++ behavior at table.cpp:353-355)
    if non_empty.is_empty() {
        return Table::from_record_batches(tables[0].ctx.clone(), tables[0].batches.clone());
    }

    // Validate all tables have the same schema
    let first_schema = non_empty[0].schema().ok_or_else(|| {
        crate::error::CylonError::new(crate::error::Code::Invalid, "First table has no schema".to_string())
    })?;

    for (i, table) in non_empty.iter().enumerate().skip(1) {
        let schema = table.schema().ok_or_else(|| {
            crate::error::CylonError::new(crate::error::Code::Invalid, format!("Table {} has no schema", i))
        })?;

        if !first_schema.eq(&schema) {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                format!("Table {} has incompatible schema", i)
            ));
        }
    }

    // Collect all batches from non-empty tables
    let mut all_batches = Vec::new();
    for table in &non_empty {
        all_batches.extend(table.batches.clone());
    }

    Table::from_record_batches(non_empty[0].ctx.clone(), all_batches)
}

/// Compare two tables for equality
/// Corresponds to C++ Equals function (table.cpp:1280-1303)
///
/// Compares two tables to check if they contain the same data.
/// If ordered=true, compares row-by-row in current order.
/// If ordered=false, sorts both tables first before comparing.
///
/// # Arguments
/// * `table_a` - First table to compare
/// * `table_b` - Second table to compare
/// * `ordered` - If true, compare in current order. If false, sort first.
///
/// # Returns
/// Returns true if tables are equal, false otherwise
///
/// # Example
/// ```ignore
/// use cylon::table::equals;
///
/// let are_equal = equals(&table1, &table2, true)?;
/// if are_equal {
///     println!("Tables are identical!");
/// }
/// ```
pub fn equals(table_a: &Table, table_b: &Table, ordered: bool) -> CylonResult<bool> {
    use arrow::compute::concat_batches;
    use crate::error::{CylonError, Code};

    // Check column count first (C++ table.cpp:1287-1289)
    if table_a.columns() != table_b.columns() {
        return Ok(false);
    }

    // Verify schemas match
    let schema_a = table_a.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "First table has no schema".to_string())
    })?;

    let schema_b = table_b.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Second table has no schema".to_string())
    })?;

    if !schema_a.eq(&schema_b) {
        return Ok(false);
    }

    if ordered {
        // Ordered comparison: compare batches directly (C++ table.cpp:1283-1284)
        // Combine batches for comparison
        let batch_a = if table_a.batches.len() > 1 {
            concat_batches(&schema_a, &table_a.batches)
                .map_err(|e| CylonError::new(Code::ExecutionError,
                    format!("Failed to combine batches from table A: {}", e)))?
        } else if table_a.batches.len() == 1 {
            table_a.batches[0].clone()
        } else {
            // Both tables empty with same schema
            return Ok(table_b.batches.is_empty());
        };

        let batch_b = if table_b.batches.len() > 1 {
            concat_batches(&schema_b, &table_b.batches)
                .map_err(|e| CylonError::new(Code::ExecutionError,
                    format!("Failed to combine batches from table B: {}", e)))?
        } else if table_b.batches.len() == 1 {
            table_b.batches[0].clone()
        } else {
            return Ok(false); // table_a has batches but table_b doesn't
        };

        // Use Arrow's RecordBatch equality
        Ok(batch_a == batch_b)
    } else {
        // Unordered comparison: sort both tables first (C++ table.cpp:1291-1299)
        // Sort on all columns in ascending order
        let num_cols = table_a.columns() as usize;
        let sort_columns: Vec<usize> = (0..num_cols).collect();
        let sort_directions = vec![true; num_cols]; // all ascending

        let sorted_a = table_a.sort_multi(&sort_columns, &sort_directions)?;
        let sorted_b = table_b.sort_multi(&sort_columns, &sort_directions)?;

        // Now compare sorted tables in ordered mode
        equals(&sorted_a, &sorted_b, true)
    }
}

/// Merge pre-sorted tables efficiently using priority queue
/// Corresponds to C++ MergeSortedTable function (table.cpp:436-497)
///
/// Merges multiple tables that are already sorted on the same columns.
/// Uses a priority queue for efficient O(N log k) merge where N is total rows
/// and k is number of tables.
///
/// # Arguments
/// * `tables` - Vector of pre-sorted tables to merge
/// * `sort_columns` - Column indices used for sorting (must match how tables were sorted)
/// * `sort_directions` - Sort direction for each column (true=ascending, false=descending)
///
/// # Returns
/// A single table containing all rows from input tables, maintaining sorted order
///
/// # Example
/// ```ignore
/// use cylon::table::merge_sorted_table;
///
/// // Assume table1, table2, table3 are already sorted by column 0
/// let merged = merge_sorted_table(&[&table1, &table2, &table3], &[0], &[true])?;
/// ```
pub fn merge_sorted_table(
    tables: &[&Table],
    sort_columns: &[usize],
    sort_directions: &[bool],
) -> CylonResult<Table> {
    use arrow::compute::concat_batches;
    use crate::error::{CylonError, Code};
    use crate::arrow::arrow_comparator::TableRowIndexEqualTo;

    if tables.is_empty() {
        return Err(CylonError::new(
            Code::Invalid,
            "Cannot merge empty vector of tables".to_string()
        ));
    }

    if sort_columns.is_empty() {
        return Err(CylonError::new(
            Code::Invalid,
            "sort_columns cannot be empty".to_string()
        ));
    }

    if sort_columns.len() != sort_directions.len() {
        return Err(CylonError::new(
            Code::Invalid,
            format!("sort_columns length {} != sort_directions length {}",
                sort_columns.len(), sort_directions.len())
        ));
    }

    // Track which rows belong to which original table (C++ table.cpp:441-447)
    let mut table_indices: Vec<i64> = Vec::new();
    let mut table_end_indices: Vec<i64> = Vec::new();
    let mut acc: i64 = 0;

    for table in tables {
        table_indices.push(acc);
        acc += table.rows() as i64;
        table_end_indices.push(acc);
    }

    // Concatenate all tables (C++ table.cpp:449)
    let concatenated = merge(tables)?;

    // TODO: When distributed context is implemented, add check here (C++ table.cpp:451-453):
    // if (concatenated->GetContext()->GetWorldSize() > 4) {
    //     return Sort(concatenated, sort_columns, out, sort_orders);
    // }
    // For now, always use priority queue merge approach

    // Combine batches for comparison (C++ table.cpp:455-456)
    let schema = concatenated.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Concatenated table has no schema".to_string())
    })?;

    let combined_batch = if concatenated.batches.len() > 1 {
        concat_batches(&schema, &concatenated.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine batches: {}", e)))?
    } else if concatenated.batches.len() == 1 {
        concatenated.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "No batches to merge".to_string()));
    };

    // Create comparator for rows with sort directions (C++ table.cpp:455-457)
    let equal_to = TableRowIndexEqualTo::new_with_columns_and_directions(
        &combined_batch,
        sort_columns,
        sort_directions
    )?;

    // Priority queue approach using custom comparison (C++ table.cpp:459-476)
    // Initialize with first row from each non-empty table
    let mut current_indices = table_indices.clone();
    let total_rows = concatenated.rows() as usize;
    let mut output_indices: Vec<i64> = Vec::with_capacity(total_rows);

    // Merge tables using priority queue logic (C++ table.cpp:468-476)
    while output_indices.len() < total_rows {
        let mut min_table: Option<usize> = None;
        let mut min_row_idx: i64 = 0;

        // Find table with minimum current row
        for (table_idx, &current_row) in current_indices.iter().enumerate() {
            if current_row < table_end_indices[table_idx] {
                if min_table.is_none() {
                    min_table = Some(table_idx);
                    min_row_idx = current_row;
                } else {
                    // Compare rows: if current row is less than min, update min
                    if equal_to.compare(current_row, min_row_idx) == std::cmp::Ordering::Less {
                        min_table = Some(table_idx);
                        min_row_idx = current_row;
                    }
                }
            }
        }

        if let Some(selected_table) = min_table {
            output_indices.push(current_indices[selected_table]);
            current_indices[selected_table] += 1;
        } else {
            break;
        }
    }

    // Use Arrow's take kernel to reorder rows (C++ table.cpp:478-481)
    let indices_array = arrow::array::Int64Array::from(output_indices);
    let mut result_columns = Vec::new();

    for col_idx in 0..combined_batch.num_columns() {
        let column = combined_batch.column(col_idx);
        let taken = arrow::compute::take(column.as_ref(), &indices_array, None)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to take rows: {}", e)))?;
        result_columns.push(taken);
    }

    let result_batch = arrow::record_batch::RecordBatch::try_new(schema, result_columns)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to create result batch: {}", e)))?;

    Table::from_record_batch(concatenated.ctx.clone(), result_batch)
}

/// Shuffle table across all processes using hash partitioning
///
/// Corresponds to C++ Shuffle() in table.cpp:1298
/// This uses ArrowAllToAll for buffer-by-buffer communication.
///
/// # Arguments
/// * `table` - Table to shuffle
/// * `hash_columns` - Column indices to hash on for partitioning
///
/// # Returns
/// Shuffled table where all rows with the same hash are on the same process
///
/// # Example
/// ```ignore
/// // Requires distributed context with MPI or FMI communicator
/// let shuffled = cylon::table::shuffle(&table, &[0])?;
/// ```
pub fn shuffle(table: &Table, hash_columns: &[usize]) -> CylonResult<Table> {
    use std::sync::{Arc, Mutex};
    use crate::error::{CylonError, Code};
    use crate::ops::partition::hash_partition_table;
    use crate::net::ops::{AllToAll, ReceiveCallback};
    use crate::arrow::arrow_all_to_all::ArrowAllToAll;

    let ctx = &table.ctx;

    if !ctx.is_distributed() {
        return Ok(table.clone());
    }

    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size();

    // Get neighbours (all processes)
    // Corresponds to C++ ctx->GetNeighbours(true)
    let neighbours: Vec<i32> = (0..world_size).collect();

    // 1. Hash partition local table
    // Corresponds to C++ shuffle_table_by_hashing -> MapToHashPartitions + Split
    let partitions = hash_partition_table(table, hash_columns, world_size as usize)?;

    // 2. Set up ArrowAllToAll
    // All processes must use the same edge_id for communication
    let edge_id = 0;
    let schema = table.schema()
        .ok_or_else(|| CylonError::new(Code::Invalid, "Table has no schema".to_string()))?;

    // Track received tables
    // Corresponds to C++ std::vector<std::shared_ptr<arrow::Table>> received_tables
    let received_tables = Arc::new(Mutex::new(Vec::new()));
    let received_tables_clone = received_tables.clone();

    // Define callback to catch receiving tables
    // Corresponds to C++ ArrowCallback (table.cpp:59-64)
    let arrow_callback = Box::new(move |_source: i32, received_table: Table, _reference: i32| {
        let mut tables = received_tables_clone.lock().unwrap();
        tables.push(received_table);
        true
    });

    // Get communicator from CylonContext and create a channel
    // Uses the generic create_channel() method (works with any communicator type)
    let communicator = ctx.get_communicator()
        .ok_or_else(|| CylonError::new(Code::Invalid, "Communicator not set".to_string()))?;

    let channel = communicator.create_channel()?;

    // Create allocator for buffers
    use crate::net::buffer::VecBuffer;
    struct SimpleAllocator;
    impl crate::net::Allocator for SimpleAllocator {
        fn allocate(&self, size: usize) -> CylonResult<Box<dyn crate::net::Buffer>> {
            Ok(Box::new(VecBuffer::new(size)))
        }
    }

    // Create ArrowAllToAll - it will create AllToAll internally
    // Matches C++ pattern: all_ = std::make_shared<AllToAll>(ctx, source, targets, edgeId, this, allocator);
    // Corresponds to C++ cylon::ArrowAllToAll (table.cpp:67-68)
    let mut arrow_all_to_all = ArrowAllToAll::new(
        rank,
        neighbours.clone(),
        neighbours.clone(),
        edge_id,
        arrow_callback,
        schema.clone(),
        ctx.clone(),
        channel,
        Box::new(SimpleAllocator),
    )?;

    // 3. Insert partitions
    // Corresponds to C++ table.cpp:70-92
    let num_partitions = partitions.len();

    if world_size as usize == num_partitions {
        // Send partitions based on index
        for (i, partition) in partitions.iter().enumerate() {
            let target = i as i32;
            if target != rank {
                if partition.num_rows() > 0 {
                    let partition_table = Table::from_record_batch(ctx.clone(), partition.clone())?;
                    arrow_all_to_all.insert(partition_table, target);
                }
            } else {
                // Keep local partition
                let mut tables = received_tables.lock().unwrap();
                let local_table = Table::from_record_batch(ctx.clone(), partition.clone())?;
                tables.push(local_table);
            }
        }
    } else {
        // Divide partitions to world_size portions
        for (i, partition) in partitions.iter().enumerate() {
            let target = (i * world_size as usize / num_partitions) as i32;
            if target != rank {
                if partition.num_rows() > 0 {
                    let partition_table = Table::from_record_batch(ctx.clone(), partition.clone())?;
                    arrow_all_to_all.insert(partition_table, target);
                }
            } else {
                let mut tables = received_tables.lock().unwrap();
                let local_table = Table::from_record_batch(ctx.clone(), partition.clone())?;
                tables.push(local_table);
            }
        }
    }

    // 4. Complete communication
    // Corresponds to C++ table.cpp:95-98
    arrow_all_to_all.finish();

    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 1000000;
    while !arrow_all_to_all.is_complete()? {
        iterations += 1;
        if iterations % 100000 == 0 {
            eprintln!("Rank {}: Shuffle iteration {}", rank, iterations);
        }
        if iterations >= MAX_ITERATIONS {
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Rank {}: Shuffle exceeded max iterations", rank),
            ));
        }
    }

    eprintln!("Rank {}: Shuffle completed after {} iterations", rank, iterations);

    arrow_all_to_all.close();

    // 5. Concatenate received tables
    // Corresponds to C++ ConcatenateTables + CombineChunks (table.cpp:104-105)
    let tables = received_tables.lock().unwrap();

    if tables.is_empty() {
        return Table::from_record_batches(ctx.clone(), Vec::new());
    }

    let mut all_batches = Vec::new();
    for table in tables.iter() {
        for batch in table.batches() {
            all_batches.push(batch.clone());
        }
    }

    Table::from_record_batches(ctx.clone(), all_batches)
}

// TODO: Port table operations from cpp/src/cylon/table.hpp:
// - FromCSV
// - WriteCSV
// - Merge
// - Join, DistributedJoin
// - DistributedUnion
// - Subtract, DistributedSubtract
// - Intersect, DistributedIntersect
// - HashPartition
// - Sort, DistributedSort
// - Select
// - Project
// - Unique, DistributedUnique
// - Slice, DistributedSlice
// - Head, Tail
// etc.