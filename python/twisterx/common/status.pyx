cdef extern from "../../../cpp/src/twisterx/status.cpp" namespace "twisterx":

    cdef enum _Code 'twisterx::Code':
        _OK 'twisterx::Code::OK'
        _OutOfMemory 'twisterx::Code::OutOfMemory'
        _KeyError 'twisterx::Code::KeyError'
        _TypeError 'twisterx::Code::TypeError'
        _Invalid 'twisterx::Code::Invalid'
        _IOError 'twisterx::Code::IOError'
        _CapacityError 'twisterx::Code::CapacityError'
        _IndexError 'twisterx::Code::IndexError'
        _UnknownError 'twisterx::Code::UnknownError'
        _NotImplemented 'twisterx::Code::NotImplemented'
        _SerializationError 'twisterx::Code::SerializationError'
        _RError 'twisterx::Code::RError'
        _CodeGenError 'twisterx::Code::CodeGenError'
        _ExpressionValidationError 'twisterx::Code::ExpressionValidationError'
        _ExecutionError 'twisterx::Code::ExecutionError'
        _AlreadyExists 'twisterx::Code::AlreadyExists'


cpdef enum Code:
    OK = _OK
    OutOfMemory = _OutOfMemory
    KeyError = _KeyError
    TypeError = _TypeError
    Invalid = _Invalid
    IOError = _IOError
    CapacityError = _CapacityError
    IndexError = _IndexError
    UnknownError = _UnknownError
    NotImplemented = _NotImplemented
    SerializationError = _SerializationError
    RError = _RError
    CodeGenError = _CodeGenError
    ExpressionValidationError = _ExpressionValidationError
    ExecutionError = _ExecutionError
    AlreadyExists = _AlreadyExists

