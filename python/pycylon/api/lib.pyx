from pycylon.data.table cimport Table
from pycylon.ctx.context cimport CCylonContextWrap
from pycylon.ctx.context cimport CylonContext
from pycylon.ctx.context import CylonContext

cdef api bint pyclon_is_context(object context):
    return isinstance(context, CylonContext)

cdef api CCylonContextWrap* pycylon_unwrap_context(object context):
    cdef CylonContext ctx
    if pyclon_is_context(context):
        ctx = <CylonContext> context
        return ctx.thisPtr
    else:
        raise ValueError('Passed object is not a CylonContext')



