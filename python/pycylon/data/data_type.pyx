from pycylon.data.data_type cimport CType
from pycylon.data.data_type cimport CLayout

cpdef enum Type:
    # Boolean as 1 bit, LSB bit-packed ordering
    BOOL = CType.ctype.CBOOL
    # Unsigned 8-bit little-endian integer
    UINT8 = CType.ctype.CUINT8
    # Signed 8-bit little-endian integer
    INT8 = CType.ctype.CINT8
    # Unsigned 16-bit little-endian integer
    UINT16 = CType.ctype.CUINT16
    # Signed 16-bit little-endian integer
    INT16 = CType.ctype.CINT16
    # Unsigned 32-bit little-endian integer
    UINT32 = CType.ctype.CUINT32
    # Signed 32-bit little-endian integer
    INT32 = CType.ctype.CINT32
    # Unsigned 64-bit little-endian integer
    UINT64 = CType.ctype.CUINT64
    # Signed 64-bit little-endian integer
    INT64 = CType.ctype.CINT64
    # 2-byte floating point value
    HALF_FLOAT = CType.ctype.CHALF_FLOAT
    # 4-byte floating point value
    FLOAT = CType.ctype.CFLOAT
    # 8-byte floating point value
    DOUBLE = CType.ctype.CDOUBLE
    # UTF8 variable-length string as List<Char>
    STRING = CType.ctype.CSTRING
    # Variable-length bytes (no guarantee of UTF8-ness)
    BINARY = CType.ctype.CBINARY
    # Fixed-size binary. Each value occupies the same number of bytes
    FIXED_SIZE_BINARY = CType.ctype.CFIXED_SIZE_BINARY
    # int32_t days since the UNIX epoch
    DATE32 = CType.ctype.CDATE32
    # int64_t milliseconds since the UNIX epoch
    DATE64 = CType.ctype.CDATE64
    # Exact timestamp encoded with int64 since UNIX epoch
    # Default unit millisecond
    TIMESTAMP = CType.ctype.CTIMESTAMP
    # Time as signed 32-bit integer, representing either seconds or
    # milliseconds since midnight
    TIME32 = CType.ctype.CTIME32
    # Time as signed 64-bit integer, representing either microseconds or
    # nanoseconds since midnight
    TIME64 = CType.ctype.CTIME64
    # YEAR_MONTH or DAY_TIME interval in SQL style
    INTERVAL = CType.ctype.CINTERVAL
    # Precision- and scale-based decimal type. Storage type depends on the
    # parameters.
    DECIMAL = CType.ctype.CDECIMAL
    # A list of some logical data type
    LIST = CType.ctype.CLIST
    # Custom data type, implemented by user
    EXTENSION = CType.ctype.CEXTENSION
    # Fixed size list of some logical type
    FIXED_SIZE_LIST = CType.ctype.CFIXED_SIZE_LIST
    # or nanoseconds.
    DURATION = CType.ctype.CDURATION


cpdef enum Layout:

    FIXED_WIDTH = CLayout.clayout.CFIXED_WIDTH
    VARIABLE_WIDTH = CLayout.clayout.CVARIABLE_WIDTH

cdef class DataType:

    def __cinit__(self, type, layout):
        if type is not None and layout is None:
            pass
