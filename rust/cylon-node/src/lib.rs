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

//! Cylon Node.js Native Addon
//!
//! This crate provides a Node.js native addon (via napi-rs) that exposes
//! Cylon's communication layer to JavaScript. It serves as the host
//! runtime for cylon-wasm distributed operations.
//!
//! Supports multiple communication backends via the Communicator trait.

#[macro_use]
extern crate napi_derive;

use napi::bindgen_prelude::*;
use std::sync::Arc;

use cylon::net::communicator::Communicator as CylonCommunicator;
use cylon::net::CommType;

#[cfg(feature = "fmi")]
use cylon::net::fmi::cylon_communicator::{FMIConfig, FMICommunicator};

#[cfg(feature = "ucx")]
use cylon::net::ucx::{UCXConfig, UCXCommunicator};

#[cfg(all(feature = "ucx", feature = "ucc"))]
use cylon::net::ucx::UCXUCCCommunicator;

#[cfg(feature = "libfabric")]
use cylon::net::libfabric::{LibfabricConfig, LibfabricCommunicator};

/// Communication backend type
#[napi(string_enum)]
pub enum CommunicatorType {
    /// FMI backend (Redis/S3 based)
    Fmi,
    /// MPI backend (requires MPI runtime)
    Mpi,
    /// Libfabric backend (high-performance fabric interface)
    Libfabric,
    /// UCX backend (Unified Communication X)
    Ucx,
    /// UCC backend (Unified Collective Communication)
    Ucc,
    /// Gloo backend (Facebook's collective communication library)
    Gloo,
}

/// Configuration options for creating an FMI communicator
#[napi(object)]
pub struct FmiConfigOptions {
    pub rank: i32,
    pub world_size: i32,
    pub host: Option<String>,
    pub port: Option<i32>,
    pub max_timeout: Option<i32>,
    pub comm_name: Option<String>,
    pub nonblocking: Option<bool>,
    pub redis_host: Option<String>,
    pub redis_port: Option<i32>,
    pub redis_namespace: Option<String>,
}

/// Configuration options for creating a UCX communicator
#[napi(object)]
pub struct UcxConfigOptions {
    /// World size (total number of processes)
    pub world_size: i32,
    /// Session ID for coordination (unique per job)
    pub session_id: String,
    /// Redis host for OOB communication (required, or set REDIS_HOST env var)
    pub redis_host: Option<String>,
    /// Redis port for OOB communication (required, or set REDIS_PORT env var)
    pub redis_port: Option<i32>,
    /// Enable UCC for collective operations (requires ucc feature)
    pub enable_ucc: Option<bool>,
}

/// Configuration options for creating a Libfabric communicator
#[napi(object)]
pub struct LibfabricConfigOptions {
    /// World size (total number of processes)
    pub world_size: i32,
    /// Session ID for coordination (unique per job)
    pub session_id: String,
    /// Redis host for OOB communication (required, or set REDIS_HOST env var)
    pub redis_host: Option<String>,
    /// Redis port for OOB communication (required, or set REDIS_PORT env var)
    pub redis_port: Option<i32>,
    /// Force specific provider (None = auto-select)
    /// Examples: "efa", "tcp", "shm", "verbs", "sockets"
    pub provider: Option<String>,
}

/// Get Redis host from config or environment variable
fn get_redis_host(config_host: Option<&str>) -> Result<String> {
    config_host
        .map(|s| s.to_string())
        .or_else(|| std::env::var("REDIS_HOST").ok())
        .ok_or_else(|| Error::from_reason(
            "Redis host not specified. Set redis_host in config or REDIS_HOST environment variable."
        ))
}

/// Get Redis port from config or environment variable
fn get_redis_port(config_port: Option<i32>) -> Result<i32> {
    config_port
        .or_else(|| std::env::var("REDIS_PORT").ok().and_then(|p| p.parse().ok()))
        .ok_or_else(|| Error::from_reason(
            "Redis port not specified. Set redis_port in config or REDIS_PORT environment variable."
        ))
}

/// Get TCPunch host from config or environment variable (for FMI)
fn get_tcpunch_host(config_host: Option<&str>) -> Result<String> {
    config_host
        .map(|s| s.to_string())
        .or_else(|| std::env::var("TCPUNCH_HOST").ok())
        .ok_or_else(|| Error::from_reason(
            "TCPunch host not specified. Set host in config or TCPUNCH_HOST environment variable."
        ))
}

/// Get TCPunch port from config or environment variable (for FMI)
fn get_tcpunch_port(config_port: Option<i32>) -> Result<i32> {
    config_port
        .or_else(|| std::env::var("TCPUNCH_PORT").ok().and_then(|p| p.parse().ok()))
        .ok_or_else(|| Error::from_reason(
            "TCPunch port not specified. Set port in config or TCPUNCH_PORT environment variable."
        ))
}

/// Generic communicator configuration
#[napi(object)]
pub struct CommunicatorConfig {
    /// Backend type to use
    pub comm_type: CommunicatorType,
    /// FMI-specific options (required if comm_type is Fmi)
    pub fmi: Option<FmiConfigOptions>,
    /// UCX-specific options (required if comm_type is Ucx or Ucc)
    pub ucx: Option<UcxConfigOptions>,
    /// Libfabric-specific options (required if comm_type is Libfabric)
    pub libfabric: Option<LibfabricConfigOptions>,
}

/// Communicator wrapper for Node.js
/// Uses the Communicator trait for backend-agnostic operations
#[napi]
pub struct Communicator {
    inner: Arc<dyn CylonCommunicator>,
}

/// Initialize the Rust logger once (idempotent).
/// Delegates to cylon::util::logging::init_logging() which uses WriteStyle::Never
/// (no ANSI color codes) and reads RUST_LOG for level — e.g. RUST_LOG=info
/// surfaces TCPunch diagnostics in CloudWatch.
fn init_logger() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        cylon::util::logging::init_logging();
    });
}

#[napi]
impl Communicator {
    /// Create a communicator with the specified backend
    #[napi(factory)]
    pub fn create(config: CommunicatorConfig) -> Result<Self> {
        init_logger();
        let inner: Arc<dyn CylonCommunicator> = match config.comm_type {
            CommunicatorType::Fmi => {
                #[cfg(feature = "fmi")]
                {
                    let fmi_opts = config.fmi.ok_or_else(|| {
                        Error::from_reason("FMI config required when using FMI backend")
                    })?;

                    let tcpunch_host = get_tcpunch_host(fmi_opts.host.as_deref())?;
                    let tcpunch_port = get_tcpunch_port(fmi_opts.port)?;
                    let redis_host = get_redis_host(fmi_opts.redis_host.as_deref())?;
                    let redis_port = get_redis_port(fmi_opts.redis_port)?;

                    let fmi_config = FMIConfig::builder()
                        .rank(fmi_opts.rank)
                        .world_size(fmi_opts.world_size)
                        .host(&tcpunch_host)
                        .port(tcpunch_port)
                        .max_timeout(fmi_opts.max_timeout.unwrap_or(30000))
                        .comm_name(fmi_opts.comm_name.as_deref().unwrap_or("cylon"))
                        .nonblocking(fmi_opts.nonblocking.unwrap_or(true))
                        .redis_host(&redis_host)
                        .redis_port(redis_port)
                        .redis_namespace(fmi_opts.redis_namespace.as_deref().unwrap_or("cylon"))
                        .build();

                    FMICommunicator::make(&fmi_config)
                        .map_err(|e| Error::from_reason(format!("Failed to create FMI communicator: {}", e)))?
                }
                #[cfg(not(feature = "fmi"))]
                {
                    return Err(Error::from_reason("FMI backend not enabled in build"));
                }
            }
            CommunicatorType::Mpi => {
                // MPI is typically not suitable for Node.js due to initialization requirements
                // but we include it for completeness
                return Err(Error::from_reason(
                    "MPI backend not supported in Node.js addon. Use FMI instead."
                ));
            }
            CommunicatorType::Libfabric => {
                #[cfg(feature = "libfabric")]
                {
                    let opts = config.libfabric.ok_or_else(|| {
                        Error::from_reason("Libfabric config required when using Libfabric backend")
                    })?;

                    let redis_host = get_redis_host(opts.redis_host.as_deref())?;
                    let redis_port = get_redis_port(opts.redis_port)? as u16;

                    let mut libfabric_config = LibfabricConfig::with_redis(
                        &redis_host,
                        redis_port,
                        &opts.session_id,
                        opts.world_size,
                    );

                    // Set provider if specified
                    if let Some(provider) = opts.provider {
                        libfabric_config.provider = Some(provider);
                    }

                    LibfabricCommunicator::new(libfabric_config)
                        .map_err(|e| Error::from_reason(format!("Failed to create Libfabric communicator: {}", e)))?
                }
                #[cfg(not(feature = "libfabric"))]
                {
                    return Err(Error::from_reason("Libfabric backend not enabled in build"));
                }
            }
            CommunicatorType::Ucx => {
                #[cfg(feature = "ucx")]
                {
                    let opts = config.ucx.ok_or_else(|| {
                        Error::from_reason("UCX config required when using UCX backend")
                    })?;

                    let redis_host = get_redis_host(opts.redis_host.as_deref())?;
                    let redis_port = get_redis_port(opts.redis_port)? as u16;

                    let ucx_config = UCXConfig::with_redis(
                        &redis_host,
                        redis_port,
                        &opts.session_id,
                        opts.world_size,
                    );

                    // Check if UCC should be enabled for collectives
                    #[cfg(feature = "ucc")]
                    if opts.enable_ucc.unwrap_or(false) {
                        let oob = cylon::net::ucx::UCXRedisOOBContext::new(&ucx_config)
                            .map_err(|e| Error::from_reason(format!("Failed to create UCX OOB context: {}", e)))?;
                        let ucx_comm = UCXCommunicator::make_oob(Box::new(oob))
                            .map_err(|e| Error::from_reason(format!("Failed to create UCX communicator: {}", e)))?;
                        Arc::new(UCXUCCCommunicator::new(ucx_comm)
                            .map_err(|e| Error::from_reason(format!("Failed to create UCX+UCC communicator: {}", e)))?)
                    } else {
                        let oob = cylon::net::ucx::UCXRedisOOBContext::new(&ucx_config)
                            .map_err(|e| Error::from_reason(format!("Failed to create UCX OOB context: {}", e)))?;
                        Arc::new(UCXCommunicator::make_oob(Box::new(oob))
                            .map_err(|e| Error::from_reason(format!("Failed to create UCX communicator: {}", e)))?)
                    }

                    #[cfg(not(feature = "ucc"))]
                    {
                        let oob = cylon::net::ucx::UCXRedisOOBContext::new(&ucx_config)
                            .map_err(|e| Error::from_reason(format!("Failed to create UCX OOB context: {}", e)))?;
                        Arc::new(UCXCommunicator::make_oob(Box::new(oob))
                            .map_err(|e| Error::from_reason(format!("Failed to create UCX communicator: {}", e)))?)
                    }
                }
                #[cfg(not(feature = "ucx"))]
                {
                    return Err(Error::from_reason("UCX backend not enabled in build"));
                }
            }
            CommunicatorType::Ucc => {
                // UCC requires UCX as the transport layer
                #[cfg(all(feature = "ucx", feature = "ucc"))]
                {
                    let opts = config.ucx.ok_or_else(|| {
                        Error::from_reason("UCX config required when using UCC backend (UCC uses UCX for transport)")
                    })?;

                    let redis_host = get_redis_host(opts.redis_host.as_deref())?;
                    let redis_port = get_redis_port(opts.redis_port)? as u16;

                    let ucx_config = UCXConfig::with_redis(
                        &redis_host,
                        redis_port,
                        &opts.session_id,
                        opts.world_size,
                    );

                    let oob = cylon::net::ucx::UCXRedisOOBContext::new(&ucx_config)
                        .map_err(|e| Error::from_reason(format!("Failed to create UCX OOB context: {}", e)))?;
                    let ucx_comm = UCXCommunicator::make_oob(Box::new(oob))
                        .map_err(|e| Error::from_reason(format!("Failed to create UCX communicator: {}", e)))?;
                    Arc::new(UCXUCCCommunicator::new(ucx_comm)
                        .map_err(|e| Error::from_reason(format!("Failed to create UCC communicator: {}", e)))?)
                }
                #[cfg(not(all(feature = "ucx", feature = "ucc")))]
                {
                    return Err(Error::from_reason("UCC backend requires both 'ucx' and 'ucc' features enabled"));
                }
            }
            CommunicatorType::Gloo => {
                // Gloo - Facebook's collective communication library
                // Not yet implemented in Cylon Rust
                return Err(Error::from_reason(
                    "Gloo backend not yet implemented in Cylon Rust. Use FMI, UCX, or Libfabric instead."
                ));
            }
        };

        Ok(Self { inner })
    }

    /// Create an FMI communicator (convenience method)
    #[napi(factory)]
    pub fn create_fmi(options: FmiConfigOptions) -> Result<Self> {
        Self::create(CommunicatorConfig {
            comm_type: CommunicatorType::Fmi,
            fmi: Some(options),
            ucx: None,
            libfabric: None,
        })
    }

    /// Create a UCX communicator (convenience method)
    #[napi(factory)]
    pub fn create_ucx(options: UcxConfigOptions) -> Result<Self> {
        Self::create(CommunicatorConfig {
            comm_type: CommunicatorType::Ucx,
            fmi: None,
            ucx: Some(options),
            libfabric: None,
        })
    }

    /// Create a Libfabric communicator (convenience method)
    #[napi(factory)]
    pub fn create_libfabric(options: LibfabricConfigOptions) -> Result<Self> {
        Self::create(CommunicatorConfig {
            comm_type: CommunicatorType::Libfabric,
            fmi: None,
            ucx: None,
            libfabric: Some(options),
        })
    }

    /// Get the communication backend type
    #[napi]
    pub fn get_comm_type(&self) -> String {
        match self.inner.get_comm_type() {
            CommType::Local => "local".to_string(),
            #[cfg(feature = "mpi")]
            CommType::Mpi => "mpi".to_string(),
            #[cfg(feature = "fmi")]
            CommType::Fmi => "fmi".to_string(),
            #[cfg(feature = "ucx")]
            CommType::Ucx => "ucx".to_string(),
            #[cfg(feature = "ucc")]
            CommType::Ucc => "ucc".to_string(),
            #[cfg(feature = "libfabric")]
            CommType::Libfabric => "libfabric".to_string(),
            #[cfg(feature = "gloo")]
            CommType::Gloo => "gloo".to_string(),
        }
    }

    #[napi]
    pub fn get_rank(&self) -> i32 {
        self.inner.get_rank()
    }

    #[napi]
    pub fn get_world_size(&self) -> i32 {
        self.inner.get_world_size()
    }

    #[napi]
    pub fn barrier(&self) -> Result<()> {
        self.inner
            .barrier()
            .map_err(|e| Error::from_reason(format!("Barrier failed: {}", e)))
    }

    /// All-to-all exchange: partitions[i] goes to worker i
    /// This is the key primitive for distributed shuffles.
    #[napi]
    pub fn all_to_all(&self, partitions: Vec<Buffer>) -> Result<Vec<Buffer>> {
        let send_data: Vec<Vec<u8>> = partitions.iter().map(|b| b.to_vec()).collect();

        let results = self
            .inner
            .all_to_all(send_data)
            .map_err(|e| Error::from_reason(format!("AllToAll failed: {}", e)))?;

        Ok(results.into_iter().map(Buffer::from).collect())
    }

    /// Allgather: each worker contributes data, all receive all data
    #[napi]
    pub fn all_gather(&self, data: Buffer) -> Result<Vec<Buffer>> {
        let results = self
            .inner
            .allgather(&data)
            .map_err(|e| Error::from_reason(format!("AllGather failed: {}", e)))?;

        Ok(results.into_iter().map(Buffer::from).collect())
    }

    /// Broadcast from root to all workers
    #[napi]
    pub fn broadcast(&self, data: Buffer, root: i32) -> Result<Buffer> {
        let mut buf = data.to_vec();
        self.inner
            .broadcast(&mut buf, root)
            .map_err(|e| Error::from_reason(format!("Broadcast failed: {}", e)))?;

        Ok(Buffer::from(buf))
    }

    /// Gather: collect data from all workers to root
    /// Implemented using allgather (as per Communicator trait design)
    /// Returns results only on root, empty vector on other workers
    #[napi]
    pub fn gather(&self, data: Buffer, root: i32) -> Result<Vec<Buffer>> {
        // Use allgather since byte-level gather isn't in the Communicator trait
        let results = self
            .inner
            .allgather(&data)
            .map_err(|e| Error::from_reason(format!("Gather failed: {}", e)))?;

        if self.inner.get_rank() == root {
            Ok(results.into_iter().map(Buffer::from).collect())
        } else {
            Ok(vec![])
        }
    }

    /// Scatter: distribute partitions from root to all workers
    /// Implemented using all_to_all (as per Communicator trait design)
    /// Root sends partition[i] to worker i, returns this worker's partition
    #[napi]
    pub fn scatter(&self, partitions: Vec<Buffer>, root: i32) -> Result<Buffer> {
        let world_size = self.inner.get_world_size() as usize;
        let rank = self.inner.get_rank();

        // Build send data: root sends partitions, others send empty
        let send_data: Vec<Vec<u8>> = if rank == root {
            if partitions.len() != world_size {
                return Err(Error::from_reason(format!(
                    "Scatter requires {} partitions, got {}",
                    world_size,
                    partitions.len()
                )));
            }
            partitions.iter().map(|b| b.to_vec()).collect()
        } else {
            vec![vec![]; world_size]
        };

        let results = self
            .inner
            .all_to_all(send_data)
            .map_err(|e| Error::from_reason(format!("Scatter failed: {}", e)))?;

        // Each worker's result is what they received from root
        Ok(Buffer::from(results[root as usize].clone()))
    }

    /// Point-to-point send
    #[napi]
    pub fn send(&self, data: Buffer, dest: i32, tag: i32) -> Result<()> {
        self.inner
            .send(&data, dest, tag)
            .map_err(|e| Error::from_reason(format!("Send failed: {}", e)))
    }

    /// Point-to-point receive
    #[napi]
    pub fn recv(&self, source: i32, tag: i32) -> Result<Buffer> {
        let mut buffer = Vec::new();
        self.inner
            .recv(&mut buffer, source, tag)
            .map_err(|e| Error::from_reason(format!("Recv failed: {}", e)))?;
        Ok(Buffer::from(buffer))
    }
}

/// Create an FMI communicator (convenience function)
#[napi]
pub fn create_communicator(options: FmiConfigOptions) -> Result<Communicator> {
    Communicator::create_fmi(options)
}
