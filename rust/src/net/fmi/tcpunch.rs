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

//! TCPunch - TCP NAT Hole Punching Library (Protocol v2)
//!
//! This module implements TCP NAT hole punching for establishing direct
//! peer-to-peer connections between nodes that may be behind NAT.
//!
//! ## Protocol v2 Changes
//!
//! - Fixed-size request (141 bytes) and response (51 bytes)
//! - Reconnection support via UUID token
//! - Explicit status codes (WAITING, PAIRED, TIMEOUT, ERROR)
//!
//! The technique works by:
//! 1. Both peers connect to a rendezvous server
//! 2. The server exchanges each peer's public IP:port information
//! 3. Both peers simultaneously attempt to connect to each other
//! 4. Using SO_REUSEADDR/SO_REUSEPORT allows binding multiple sockets to the same port
//! 5. One connection succeeds (either active connect or passive accept)

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, SocketAddr, Ipv4Addr};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{CylonError, CylonResult, Code};

// ============================================================================
// Protocol v2 Constants
// ============================================================================

/// Maximum length of pairing name
pub const MAX_PAIRING_NAME: usize = 100;

/// Length of reconnection token (UUID string)
pub const TOKEN_LENGTH: usize = 37;

/// Client request size (100 + 37 + 4 = 141 bytes)
pub const CLIENT_REQUEST_SIZE: usize = 141;

/// Server response size — matches C++ SERVER_RESPONSE_SIZE = 50 bytes.
/// Layout: status(1) + your_ip(4) + your_port(2) + peer_ip(4) + peer_port(2) + token(37) = 50.
/// The parser only reads through offset 49; the earlier Rust value of 51 was off-by-one.
pub const SERVER_RESPONSE_SIZE: usize = 50;

/// Magic number for validation handshake
const VALIDATION_MAGIC: u32 = 0xDEADBEEF;

/// Default timeout for connection attempts (30 seconds)
const DEFAULT_TIMEOUT_MS: u64 = 30000;

/// Validation timeout (15 seconds)
const VALIDATION_TIMEOUT_SECS: u64 = 15;

/// Default max retries for reconnection
const DEFAULT_MAX_RETRIES: u32 = 3;

/// TCP keepalive time (start probing after this many seconds of idle)
const KEEPALIVE_TIME_SECS: u64 = 5;

/// TCP keepalive interval (seconds between probes)
const KEEPALIVE_INTERVAL_SECS: u64 = 2;

/// TCP keepalive retry count (consider dead after this many failed probes)
const KEEPALIVE_RETRIES: u32 = 3;

// ============================================================================
// Protocol v2 Types
// ============================================================================

/// Pairing status returned by server
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PairingStatus {
    /// Registered, waiting for peer
    Waiting = 0,
    /// Peer found, proceed to hole punching
    Paired = 1,
    /// Server-side timeout, reconnect with token
    Timeout = 2,
    /// Invalid request/token, start fresh
    Error = 3,
}

impl From<u8> for PairingStatus {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Waiting,
            1 => Self::Paired,
            2 => Self::Timeout,
            _ => Self::Error,
        }
    }
}

/// Peer information (IP and port)
#[derive(Debug, Clone, Copy)]
pub struct PeerInfo {
    pub ip: Ipv4Addr,
    pub port: u16,
}

impl PeerInfo {
    pub fn to_socket_addr(&self) -> SocketAddr {
        SocketAddr::new(std::net::IpAddr::V4(self.ip), self.port)
    }

    pub fn is_empty(&self) -> bool {
        self.ip.is_unspecified() && self.port == 0
    }
}

/// Server response (51 bytes)
#[derive(Debug)]
pub struct ServerResponse {
    pub status: PairingStatus,
    pub your_info: PeerInfo,
    pub peer_info: Option<PeerInfo>,
    pub token: String,
}

/// Legacy peer connection data (for compatibility)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PeerConnectionData {
    pub ip: u32,        // IPv4 address in network byte order
    pub port: u16,      // Port in network byte order
}

/// Validation message for handshake
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ValidationMsg {
    pub magic: u32,
    pub peer_id: u32,
    pub timestamp: u32,
}

impl ValidationMsg {
    pub fn new() -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        Self {
            magic: VALIDATION_MAGIC,
            peer_id: 0,
            timestamp,
        }
    }

    pub fn to_bytes(&self) -> [u8; 12] {
        let mut bytes = [0u8; 12];
        bytes[0..4].copy_from_slice(&self.magic.to_ne_bytes());
        bytes[4..8].copy_from_slice(&self.peer_id.to_ne_bytes());
        bytes[8..12].copy_from_slice(&self.timestamp.to_ne_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8; 12]) -> Self {
        Self {
            magic: u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            peer_id: u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            timestamp: u32::from_ne_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        }
    }
}

impl Default for ValidationMsg {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerConnectionData {
    pub fn to_bytes(&self) -> [u8; 6] {
        let mut bytes = [0u8; 6];
        bytes[0..4].copy_from_slice(&self.ip.to_ne_bytes());
        bytes[4..6].copy_from_slice(&self.port.to_ne_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8; 6]) -> Self {
        Self {
            ip: u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            port: u16::from_ne_bytes([bytes[4], bytes[5]]),
        }
    }

    pub fn to_socket_addr(&self) -> SocketAddr {
        let ip = Ipv4Addr::from(u32::from_be(self.ip));
        let port = u16::from_be(self.port);
        SocketAddr::new(std::net::IpAddr::V4(ip), port)
    }
}

// ============================================================================
// Protocol v2 Functions
// ============================================================================

/// Build a client request (141 bytes)
pub fn build_request(pairing_name: &str, token: Option<&str>) -> [u8; CLIENT_REQUEST_SIZE] {
    let mut buf = [0u8; CLIENT_REQUEST_SIZE];

    // Write pairing name (offset 0, max 99 chars + null)
    let name_bytes = pairing_name.as_bytes();
    let len = name_bytes.len().min(MAX_PAIRING_NAME - 1);
    buf[..len].copy_from_slice(&name_bytes[..len]);

    // Write reconnect token if present (offset 100, max 36 chars + null)
    if let Some(t) = token {
        let token_bytes = t.as_bytes();
        let len = token_bytes.len().min(TOKEN_LENGTH - 1);
        buf[MAX_PAIRING_NAME..MAX_PAIRING_NAME + len].copy_from_slice(&token_bytes[..len]);
    }

    // Flags at offset 137 (4 bytes) - reserved, set to 0
    // Already zero-initialized

    buf
}

/// Parse server response (51 bytes)
pub fn parse_response(buf: &[u8; SERVER_RESPONSE_SIZE]) -> ServerResponse {
    // Status (1 byte at offset 0)
    let status = PairingStatus::from(buf[0]);

    // Your IP (4 bytes at offset 1, network byte order)
    let your_ip = Ipv4Addr::new(buf[1], buf[2], buf[3], buf[4]);
    // Your port (2 bytes at offset 5, network byte order)
    let your_port = u16::from_be_bytes([buf[5], buf[6]]);

    // Peer IP (4 bytes at offset 7, network byte order)
    let peer_ip = Ipv4Addr::new(buf[7], buf[8], buf[9], buf[10]);
    // Peer port (2 bytes at offset 11, network byte order)
    let peer_port = u16::from_be_bytes([buf[11], buf[12]]);

    // Token (37 bytes at offset 13)
    let token_end = buf[13..50].iter().position(|&b| b == 0).unwrap_or(37);
    let token = String::from_utf8_lossy(&buf[13..13 + token_end]).to_string();

    // Determine if peer info is valid
    let peer_info = if peer_ip.is_unspecified() && peer_port == 0 {
        None
    } else {
        Some(PeerInfo { ip: peer_ip, port: peer_port })
    };

    ServerResponse {
        status,
        your_info: PeerInfo { ip: your_ip, port: your_port },
        peer_info,
        token,
    }
}

// ============================================================================
// Socket Configuration
// ============================================================================

/// Configure socket with reuse options
#[cfg(unix)]
fn configure_socket_reuse(socket: &socket2::Socket) -> CylonResult<()> {
    socket.set_reuse_address(true).map_err(|e| {
        CylonError::new(Code::IoError, format!("Failed to set SO_REUSEADDR: {}", e))
    })?;

    // SO_REUSEPORT is available on Linux and most Unix platforms
    // socket2 provides this on all Unix platforms
    #[cfg(all(unix, not(any(target_os = "solaris", target_os = "illumos"))))]
    {
        socket.set_reuse_port(true).map_err(|e| {
            CylonError::new(Code::IoError, format!("Failed to set SO_REUSEPORT: {}", e))
        })?;
    }

    Ok(())
}

/// Configure TCP keepalive on a stream for fast failure detection
///
/// This is critical for serverless environments where peers can die unexpectedly.
/// With these aggressive settings, a dead peer will be detected within ~11 seconds:
/// - 5 seconds idle before first probe
/// - 3 probes at 2 second intervals
///
/// Without keepalive, the connection could appear alive for minutes.
pub fn configure_keepalive(stream: &TcpStream) -> CylonResult<()> {
    use socket2::SockRef;

    let socket = SockRef::from(stream);

    // Enable keepalive
    socket.set_keepalive(true).map_err(|e| {
        CylonError::new(Code::IoError, format!("Failed to enable keepalive: {}", e))
    })?;

    // Set keepalive parameters using socket2's TcpKeepalive
    let keepalive = socket2::TcpKeepalive::new()
        .with_time(Duration::from_secs(KEEPALIVE_TIME_SECS));

    // On Linux, we can also set interval and retries
    #[cfg(target_os = "linux")]
    let keepalive = keepalive
        .with_interval(Duration::from_secs(KEEPALIVE_INTERVAL_SECS))
        .with_retries(KEEPALIVE_RETRIES);

    // On macOS, we can set interval but not retries
    #[cfg(target_os = "macos")]
    let keepalive = keepalive
        .with_interval(Duration::from_secs(KEEPALIVE_INTERVAL_SECS));

    socket.set_tcp_keepalive(&keepalive).map_err(|e| {
        CylonError::new(Code::IoError, format!("Failed to set keepalive params: {}", e))
    })?;

    log::debug!(
        "Configured TCP keepalive: time={}s, interval={}s, retries={}",
        KEEPALIVE_TIME_SECS,
        KEEPALIVE_INTERVAL_SECS,
        KEEPALIVE_RETRIES
    );

    Ok(())
}

/// Configure TCP keepalive with custom parameters
///
/// # Arguments
/// * `stream` - The TCP stream to configure
/// * `time_secs` - Seconds of idle before first probe
/// * `interval_secs` - Seconds between probes
/// * `retries` - Number of probes before considering dead (Linux only)
pub fn configure_keepalive_custom(
    stream: &TcpStream,
    time_secs: u64,
    interval_secs: u64,
    #[allow(unused_variables)] retries: u32,
) -> CylonResult<()> {
    use socket2::SockRef;

    let socket = SockRef::from(stream);

    socket.set_keepalive(true).map_err(|e| {
        CylonError::new(Code::IoError, format!("Failed to enable keepalive: {}", e))
    })?;

    let keepalive = socket2::TcpKeepalive::new()
        .with_time(Duration::from_secs(time_secs));

    #[cfg(target_os = "linux")]
    let keepalive = keepalive
        .with_interval(Duration::from_secs(interval_secs))
        .with_retries(retries);

    #[cfg(target_os = "macos")]
    let keepalive = keepalive
        .with_interval(Duration::from_secs(interval_secs));

    socket.set_tcp_keepalive(&keepalive).map_err(|e| {
        CylonError::new(Code::IoError, format!("Failed to set keepalive params: {}", e))
    })?;

    Ok(())
}

/// Listener thread — exact Rust port of C++ peer_listen().
///
/// C++ uses a BLOCKING socket with SO_RCVTIMEO=1s so accept() times out every
/// second and re-checks connection_established.  Do NOT set non-blocking here —
/// it would override SO_RCVTIMEO and cause accept() to return WouldBlock
/// immediately every call, which is functionally different and causes the
/// listener to spin rather than wait for the kernel to deliver an incoming SYN.
fn peer_listen(
    local_port: u16,
    connection_established: Arc<AtomicBool>,
    accepting_socket: Arc<AtomicI32>,
    listener_ready: Arc<AtomicBool>,
) -> CylonResult<()> {
    use socket2::{Domain, Protocol, Socket, Type};

    log::info!("peer_listen: creating listener on port {}", local_port);

    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
        .map_err(|e| CylonError::new(Code::IoError, format!("peer_listen: socket create failed: {}", e)))?;

    // SO_REUSEADDR + SO_REUSEPORT — matches C++ peer_listen lines 96-101
    configure_socket_reuse(&socket)?;
    log::info!("peer_listen: SO_REUSEADDR+SO_REUSEPORT set on port {}", local_port);

    // SO_RCVTIMEO = 1 second, BLOCKING — matches C++ peer_listen lines 105-112.
    // This makes accept() block for up to 1 second then return EAGAIN, allowing
    // the loop to check connection_established frequently.
    // C++ does NOT set O_NONBLOCK on the listen socket.
    socket.set_read_timeout(Some(Duration::from_secs(1)))
        .map_err(|e| CylonError::new(Code::IoError, format!("peer_listen: SO_RCVTIMEO failed: {}", e)))?;

    // Bind — matches C++ peer_listen lines 114-123
    let addr: SocketAddr = format!("0.0.0.0:{}", local_port).parse().unwrap();
    match socket.bind(&addr.into()) {
        Ok(_) => log::info!("peer_listen: bound to 0.0.0.0:{}", local_port),
        Err(e) => {
            log::error!("peer_listen: bind to port {} failed: {} (errno={:?})", local_port, e, e.raw_os_error());
            return Err(CylonError::new(Code::IoError, format!("peer_listen: bind failed: {}", e)));
        }
    }

    // Listen — matches C++ peer_listen lines 125-129
    socket.listen(1)
        .map_err(|e| CylonError::new(Code::IoError, format!("peer_listen: listen failed: {}", e)))?;
    log::info!("peer_listen: listening on port {} (BLOCKING, SO_RCVTIMEO=1s)", local_port);

    // Signal ready — listener is bound and listening before connect loop starts
    listener_ready.store(true, Ordering::SeqCst);

    // Convert to TcpListener — stays BLOCKING with 1s read timeout (matches C++)
    // Do NOT call set_nonblocking(true) — that is NOT in the C++ code
    let listener: TcpListener = socket.into();

    let mut error_count = 0;

    // Accept loop — exact match of C++ peer_listen lines 135-166
    loop {
        if connection_established.load(Ordering::SeqCst) {
            break;
        }

        match listener.accept() {
            Ok((stream, peer_addr)) => {
                log::info!("peer_listen: accepted connection from {}", peer_addr);
                #[cfg(unix)]
                {
                    use std::os::unix::io::AsRawFd;
                    accepting_socket.store(stream.as_raw_fd(), Ordering::SeqCst);
                    std::mem::forget(stream);
                }
                connection_established.store(true, Ordering::SeqCst);
                return Ok(());
            }
            // SO_RCVTIMEO expired (EAGAIN/EWOULDBLOCK/TimedOut) — loop and re-check flag
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock
                       || e.kind() == std::io::ErrorKind::TimedOut => {
                continue; // matches C++ "if (errno == EAGAIN || EWOULDBLOCK) continue;"
            }
            Err(e) => {
                log::warn!("peer_listen: accept error: {} (os={:?})", e, e.raw_os_error());
                error_count += 1;
                if error_count > 5 {
                    let backoff = std::cmp::min(100 * (1 << (error_count - 5)), 5000);
                    thread::sleep(Duration::from_millis(backoff));
                }
            }
        }
    }

    Ok(())
}

/// Core hole-punch logic using pre-created listener state.
/// The listener thread must already be spawned and its handle passed in.
/// This allows the WAITING path to start the listener before blocking on the
/// second rendezvous response (matching C++ behaviour), while the PAIRED path
/// creates fresh state and spawns the listener just before calling this.
fn do_hole_punch_inner(
    your_info: &PeerInfo,
    peer_info: &PeerInfo,
    timeout_ms: u64,
    connection_established: Arc<AtomicBool>,
    accepting_socket: Arc<AtomicI32>,
    listener_ready: Arc<AtomicBool>,
    listener_handle: thread::JoinHandle<CylonResult<()>>,
) -> CylonResult<TcpStream> {
    use socket2::{Domain, Protocol, Socket, Type};

    let local_port = your_info.port;
    let peer_addr = peer_info.to_socket_addr();

    log::info!("do_hole_punch_inner: your_port={} peer={}", local_port, peer_addr);

    // Create peer socket — SO_REUSEADDR + SO_REUSEPORT + NON-BLOCKING.
    // Matches C++ do_hole_punch lines 182-191.
    let peer_socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
        .map_err(|e| CylonError::new(Code::IoError, format!("peer socket create failed: {}", e)))?;

    configure_socket_reuse(&peer_socket)?;

    // Set NON-BLOCKING on the ACTIVE (connect) socket only — matches C++ line 189.
    // The LISTENER socket is kept BLOCKING with SO_RCVTIMEO (see peer_listen).
    peer_socket.set_nonblocking(true)
        .map_err(|e| CylonError::new(Code::IoError, format!("peer socket set_nonblocking failed: {}", e)))?;
    log::info!("do_hole_punch_inner: peer socket created, SO_REUSEADDR+SO_REUSEPORT+NONBLOCKING");

    // Bail macro: signal listener to stop and join before any early return
    macro_rules! bail {
        ($err:expr) => {{
            connection_established.store(true, Ordering::SeqCst);
            let _ = listener_handle.join();
            return Err($err);
        }};
    }

    // Bind peer socket to same local port as rendezvous — matches C++ line 196-204.
    // Both listener and peer socket share this port via SO_REUSEPORT.
    let local_addr: SocketAddr = format!("0.0.0.0:{}", local_port).parse().unwrap();
    match peer_socket.bind(&local_addr.into()) {
        Ok(_) => log::info!("do_hole_punch_inner: peer socket bound to 0.0.0.0:{}", local_port),
        Err(e) => {
            log::error!("do_hole_punch_inner: peer socket bind to {} failed: {} (errno={:?})",
                local_port, e, e.raw_os_error());
            bail!(CylonError::new(Code::IoError, format!("peer socket bind failed: {}", e)));
        }
    }

    // Wait for listener to bind and listen before starting connect attempts.
    // C++ avoids this race by spawning the listener before calling do_hole_punch().
    // Without this wait, connect() can start before the LISTEN socket is ready,
    // so the peer's incoming SYN has nowhere to land and gets dropped.
    let wait_start = Instant::now();
    while !listener_ready.load(Ordering::SeqCst) {
        if wait_start.elapsed() > Duration::from_secs(5) {
            log::warn!("Listener did not become ready within 5s — proceeding anyway");
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }
    log::info!("Listener ready after {}ms", wait_start.elapsed().as_millis());

    // Active connect loop — exact Rust port of C++ do_hole_punch lines 215-247.
    // C++ does NOT sleep on EINPROGRESS/EALREADY — it tight-loops.
    // Sleeping 10ms here (as Rust did before) delays SYN delivery and breaks
    // the simultaneous-open timing that TCPunch requires.
    log::info!("do_hole_punch: starting connect loop to {} from local port {} (timeout={}ms)",
               peer_addr, local_port, timeout_ms);
    let start_time = Instant::now();
    let max_connection_time = Duration::from_millis(timeout_ms);
    let mut attempt_count = 0u64;
    let mut connected = false;

    while !connection_established.load(Ordering::SeqCst) {
        if start_time.elapsed() > max_connection_time {
            bail!(CylonError::new(Code::IoError,
                format!("Connection timeout after {}ms, {} attempts", timeout_ms, attempt_count)));
        }

        match peer_socket.connect(&peer_addr.into()) {
            Ok(()) => {
                log::info!("do_hole_punch: connect() succeeded after {} attempts", attempt_count);
                connected = true;
                break;
            }
            Err(ref e) if e.raw_os_error() == Some(libc::EISCONN) => {
                log::info!("do_hole_punch: EISCONN (connected) after {} attempts", attempt_count);
                connected = true;
                break;
            }
            // EINPROGRESS/EALREADY/EAGAIN — matches C++ "continue" with NO sleep (tight loop)
            Err(ref e) if e.raw_os_error() == Some(libc::EALREADY)
                       || e.raw_os_error() == Some(libc::EAGAIN)
                       || e.raw_os_error() == Some(libc::EINPROGRESS) => {
                attempt_count += 1;
                // C++ does NOT sleep here — tight polling loop
                continue;
            }
            Err(ref e) => {
                // Log every error with full details for triage
                log::warn!("do_hole_punch: connect attempt {} errno={:?} kind={:?} elapsed={}ms",
                    attempt_count, e.raw_os_error(), e.kind(), start_time.elapsed().as_millis());
                let base_delay = 100u64;
                let backoff_delay = base_delay * (1 + attempt_count / 10);
                thread::sleep(Duration::from_millis(std::cmp::min(backoff_delay, 1000)));
                attempt_count += 1;
                continue;
            }
        }
    }

    // Determine which connection to use
    let mut peer_stream = if connection_established.load(Ordering::SeqCst) && !connected {
        // Listener accepted connection
        let _ = listener_handle.join();
        let fd = accepting_socket.load(Ordering::SeqCst);
        if fd < 0 {
            return Err(CylonError::new(Code::IoError, "No valid socket from listener".to_string()));
        }

        #[cfg(unix)]
        unsafe {
            use std::os::unix::io::FromRawFd;
            TcpStream::from_raw_fd(fd)
        }
        #[cfg(not(unix))]
        {
            return Err(CylonError::new(Code::IoError, "Platform not supported".to_string()));
        }
    } else {
        // Active connection succeeded
        connection_established.store(true, Ordering::SeqCst);
        let _ = listener_handle.join();

        peer_socket.set_nonblocking(false)
            .map_err(|e| CylonError::new(Code::IoError,
                format!("Failed to set blocking: {}", e)))?;
        peer_socket.into()
    };

    // Perform validation handshake
    peer_stream.set_read_timeout(Some(Duration::from_secs(VALIDATION_TIMEOUT_SECS)))
        .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set timeout: {}", e)))?;
    peer_stream.set_write_timeout(Some(Duration::from_secs(VALIDATION_TIMEOUT_SECS)))
        .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set timeout: {}", e)))?;

    // Send validation message
    let validation_msg = ValidationMsg::new();
    peer_stream.write_all(&validation_msg.to_bytes())
        .map_err(|e| {
            log::error!("Validation handshake failed: could not send validation message: {}", e);
            CylonError::new(Code::IoError, "Validation handshake failed: send".to_string())
        })?;

    // Receive peer's validation message
    let mut peer_validation_bytes = [0u8; 12];
    peer_stream.read_exact(&mut peer_validation_bytes)
        .map_err(|e| {
            log::error!("Validation handshake failed: could not receive validation: {}", e);
            CylonError::new(Code::IoError, "Validation handshake failed: receive".to_string())
        })?;

    let peer_validation = ValidationMsg::from_bytes(&peer_validation_bytes);
    if peer_validation.magic != VALIDATION_MAGIC {
        log::error!("Validation handshake failed: invalid magic number");
        return Err(CylonError::new(Code::IoError,
            "Validation handshake failed: invalid magic".to_string()));
    }

    log::info!("Validation handshake completed successfully");

    // Configure TCP keepalive for fast failure detection
    // This is critical for serverless environments where peers can die unexpectedly
    if let Err(e) = configure_keepalive(&peer_stream) {
        log::warn!("Failed to configure TCP keepalive (non-fatal): {}", e);
    }

    // Clear timeouts for normal operation
    // Note: With keepalive enabled, dead peers will be detected within ~11 seconds
    peer_stream.set_read_timeout(None).ok();
    peer_stream.set_write_timeout(None).ok();

    Ok(peer_stream)
}

/// Perform hole punching for the PAIRED path — creates fresh listener state and
/// spawns the listener thread, then delegates to do_hole_punch_inner.
fn do_hole_punch(
    your_info: &PeerInfo,
    peer_info: &PeerInfo,
    timeout_ms: u64,
) -> CylonResult<TcpStream> {
    let local_port = your_info.port;
    let connection_established = Arc::new(AtomicBool::new(false));
    let accepting_socket = Arc::new(AtomicI32::new(-1));
    let listener_ready = Arc::new(AtomicBool::new(false));
    let wce = connection_established.clone();
    let was = accepting_socket.clone();
    let wlr = listener_ready.clone();
    let handle = thread::spawn(move || peer_listen(local_port, wce, was, wlr));
    do_hole_punch_inner(your_info, peer_info, timeout_ms,
                        connection_established, accepting_socket, listener_ready, handle)
}

/// Perform hole punching using pre-created listener state (WAITING path).
/// The listener thread was already spawned before waiting for the second
/// rendezvous response, matching C++ behaviour.
fn do_hole_punch_with_listener(
    your_info: &PeerInfo,
    peer_info: &PeerInfo,
    timeout_ms: u64,
    connection_established: Arc<AtomicBool>,
    accepting_socket: Arc<AtomicI32>,
    listener_ready: Arc<AtomicBool>,
) -> CylonResult<TcpStream> {
    // The listener thread is already running; we need its handle to join it.
    // Since we can't pass the handle through the WAITING-path Arc sharing, we
    // re-use do_hole_punch_inner directly with the pre-created arcs. The handle
    // is held by the WAITING path caller which joins it unconditionally.
    // We create a dummy no-op thread here just to satisfy the signature;
    // the real listener handle is joined by the caller.
    let dummy_handle = thread::spawn(|| Ok(()));
    do_hole_punch_inner(your_info, peer_info, timeout_ms,
                        connection_established, accepting_socket, listener_ready, dummy_handle)
}

/// Establish a peer-to-peer connection using TCP NAT hole punching (Protocol v2)
///
/// # Arguments
/// * `pairing_name` - Unique name for this pairing (both peers must use the same name)
/// * `server_address` - IP address of the rendezvous server
/// * `port` - Port of the rendezvous server (default: 10000)
/// * `timeout_ms` - Connection timeout in milliseconds (0 for default 30s)
///
/// # Returns
/// * `Ok(TcpStream)` - Successfully established connection
/// * `Err(CylonError)` - Connection failed (timeout, validation failure, etc.)
pub fn pair(
    pairing_name: &str,
    server_address: &str,
    port: u16,
    timeout_ms: u64,
) -> CylonResult<TcpStream> {
    pair_with_retries(pairing_name, server_address, port, timeout_ms, DEFAULT_MAX_RETRIES)
}

/// Establish a peer-to-peer connection with configurable retries (Protocol v2)
///
/// # Arguments
/// * `pairing_name` - Unique name for this pairing (both peers must use the same name)
/// * `server_address` - IP address of the rendezvous server
/// * `port` - Port of the rendezvous server
/// * `timeout_ms` - Connection timeout in milliseconds (0 for default 30s)
/// * `max_retries` - Maximum number of reconnection attempts
///
/// # Returns
/// * `Ok(TcpStream)` - Successfully established connection
/// * `Err(CylonError)` - Connection failed after all retries
pub fn pair_with_retries(
    pairing_name: &str,
    server_address: &str,
    port: u16,
    timeout_ms: u64,
    max_retries: u32,
) -> CylonResult<TcpStream> {
    use socket2::{Domain, Protocol, Socket, Type};

    let timeout_ms = if timeout_ms == 0 { DEFAULT_TIMEOUT_MS } else { timeout_ms };
    let timeout = Duration::from_millis(timeout_ms);

    // Resolve rendezvous address — supports both IP literals and DNS hostnames.
    use std::net::ToSocketAddrs;
    let server_addr: SocketAddr = format!("{}:{}", server_address, port)
        .to_socket_addrs()
        .map_err(|e| CylonError::new(Code::IoError, format!("Failed to resolve rendezvous '{}:{}': {}", server_address, port, e)))?
        .find(|a| a.is_ipv4())
        .ok_or_else(|| CylonError::new(Code::IoError, format!("No IPv4 address for rendezvous '{}'", server_address)))?;
    log::info!("Resolved rendezvous '{}:{}' → {}", server_address, port, server_addr);

    let mut reconnect_token: Option<String> = None;

    for attempt in 0..max_retries {
        log::debug!("Pairing attempt {} of {} for '{}'", attempt + 1, max_retries, pairing_name);

        // Connect to rendezvous server
        let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
            .map_err(|e| CylonError::new(Code::IoError, format!("Socket creation failed: {}", e)))?;

        configure_socket_reuse(&socket)?;

        socket.set_read_timeout(Some(timeout))
            .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set timeout: {}", e)))?;
        socket.set_write_timeout(Some(timeout))
            .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set timeout: {}", e)))?;

        socket.connect(&server_addr.into())
            .map_err(|e| CylonError::new(Code::IoError,
                format!("Connection to rendezvous server failed: {}", e)))?;

        let mut stream: TcpStream = socket.into();

        // Send request (141 bytes)
        let request = build_request(pairing_name, reconnect_token.as_deref());
        stream.write_all(&request)
            .map_err(|e| CylonError::new(Code::IoError,
                format!("Failed to send request: {}", e)))?;

        // Receive response (51 bytes)
        let mut resp_buf = [0u8; SERVER_RESPONSE_SIZE];
        stream.read_exact(&mut resp_buf)
            .map_err(|e| CylonError::new(Code::IoError,
                format!("Failed to receive response: {}", e)))?;

        let resp = parse_response(&resp_buf);

        // Save token for potential reconnection
        if !resp.token.is_empty() {
            reconnect_token = Some(resp.token.clone());
        }

        match resp.status {
            PairingStatus::Paired => {
                // Got peer immediately
                let peer = resp.peer_info.ok_or_else(|| {
                    CylonError::new(Code::IoError, "No peer info in PAIRED response".to_string())
                })?;

                // Use server-reported your_port — exactly like C++ (public_info.port = resp.your_port).
                // This is the external NAT port the peer knows about and will connect to.
                let your_port = resp.your_info.port;
                log::info!("pair PAIRED: your_port={} (server-reported), peer={}:{}",
                    your_port, peer.ip, peer.port);
                // Keep stream alive during hole punch — preserves NAT entry (matches C++)
                let result = do_hole_punch(&resp.your_info, &peer, timeout_ms);
                drop(stream);
                return result;
            }

            PairingStatus::Waiting => {
                log::debug!("Registered, waiting for peer (token: {})", resp.token);

                // C++ starts the listener thread HERE — before blocking on the second
                // rendezvous response — so it is already bound and listening by the
                // time the peer info arrives and do_hole_punch() begins.
                // Rust previously started the listener only inside do_hole_punch(),
                // which is too late: the peer's SYN could arrive before the LISTEN
                // socket exists.
                // Use server-reported your_port — same as C++ (matches PAIRED path above)
                let your_port_waiting = resp.your_info.port;
                log::info!("pair WAITING: your_port={} (server-reported)", your_port_waiting);

                let wait_conn_established = Arc::new(AtomicBool::new(false));
                let wait_accepting_socket = Arc::new(AtomicI32::new(-1));
                let wait_listener_ready = Arc::new(AtomicBool::new(false));
                let wce = wait_conn_established.clone();
                let was = wait_accepting_socket.clone();
                let wlr = wait_listener_ready.clone();
                let listener_handle_waiting = thread::spawn(move || {
                    peer_listen(your_port_waiting, wce, was, wlr)
                });

                // Wait for second response with peer info
                let hole_punch_result = match stream.read_exact(&mut resp_buf) {
                    Ok(()) => {
                        let resp2 = parse_response(&resp_buf);
                        if resp2.status == PairingStatus::Paired {
                            let peer = resp2.peer_info.ok_or_else(|| {
                                CylonError::new(Code::IoError, "No peer info in PAIRED response".to_string())
                            });
                            match peer {
                                Ok(peer) => {
                                    log::info!("Peer found: {}:{} (your_port={}) conn_established={}",
                                               peer.ip, peer.port, your_port_waiting,
                                               wait_conn_established.load(Ordering::SeqCst));
                                    // Do NOT reset connection_established here — if the listener already
                                    // accepted the peer's incoming SYN while we were waiting for the
                                    // second rendezvous response, resetting to false would discard that
                                    // accepted socket and cause the connect loop to run forever.
                                    // (C++ never resets connection_established between listener spawn
                                    //  and do_hole_punch; the flag flows through continuously.)
                                    let result = do_hole_punch_with_listener(
                                        &resp.your_info, &peer, timeout_ms,
                                        wait_conn_established.clone(),
                                        wait_accepting_socket.clone(),
                                        wait_listener_ready.clone(),
                                    );
                                    drop(stream);
                                    Some(result)
                                }
                                Err(e) => { Some(Err(e)) }
                            }
                        } else {
                            log::warn!("Unexpected status after WAITING: {:?}", resp2.status);
                            None
                        }
                    }
                    Err(e) => {
                        log::warn!("Timeout waiting for peer (attempt {}): {}", attempt + 1, e);
                        None
                    }
                };

                // Always join the pre-started listener thread
                wait_conn_established.store(true, Ordering::SeqCst);
                let _ = listener_handle_waiting.join();

                if let Some(result) = hole_punch_result {
                    return result;
                }
                // Fall through to retry
            }

            PairingStatus::Timeout => {
                log::warn!("Server timeout (attempt {}), will retry with token", attempt + 1);
                // Token is already saved, retry
                thread::sleep(Duration::from_millis(1000));
                continue;
            }

            PairingStatus::Error => {
                log::warn!("Server error (attempt {}), clearing token and retrying", attempt + 1);
                reconnect_token = None;
                thread::sleep(Duration::from_millis(1000));
                continue;
            }
        }
    }

    Err(CylonError::new(
        Code::IoError,
        format!("Failed to pair '{}' after {} retries", pairing_name, max_retries),
    ))
}

/// Remove a pairing from the rendezvous server (Protocol v2)
///
/// Note: In Protocol v2, there is no explicit "remove" operation.
/// The server automatically cleans up pairings after timeout.
/// This function sends a request with an empty token which effectively
/// creates a new registration that will timeout if no peer connects.
///
/// For immediate cleanup, clients should simply disconnect and let
/// the server handle cleanup via its internal timeout mechanism.
pub fn remove_pair(
    pairing_name: &str,
    server_address: &str,
    port: u16,
    timeout_ms: u64,
) -> CylonResult<()> {
    let timeout = Duration::from_millis(if timeout_ms == 0 { DEFAULT_TIMEOUT_MS } else { timeout_ms });

    use std::net::ToSocketAddrs;
    let server_addr: SocketAddr = format!("{}:{}", server_address, port)
        .to_socket_addrs()
        .map_err(|e| CylonError::new(Code::IoError, format!("Failed to resolve '{}:{}': {}", server_address, port, e)))?
        .find(|a| a.is_ipv4())
        .ok_or_else(|| CylonError::new(Code::IoError, format!("No IPv4 address for '{}'", server_address)))?;

    let mut stream = TcpStream::connect_timeout(&server_addr, timeout)
        .map_err(|e| CylonError::new(Code::IoError,
            format!("Connection to rendezvous server failed: {}", e)))?;

    stream.set_write_timeout(Some(timeout))
        .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set timeout: {}", e)))?;

    // Send a Protocol v2 request (141 bytes) with no reconnect token
    // The server will register this and clean it up after timeout
    let request = build_request(pairing_name, None);
    stream.write_all(&request)
        .map_err(|e| CylonError::new(Code::IoError,
            format!("Failed to send request: {}", e)))?;

    // Immediately close the connection - server will clean up
    drop(stream);

    log::debug!("Sent remove request for pairing '{}'", pairing_name);

    Ok(())
}
