from pycylon.data.table cimport Table
from pycylon.ctx.context cimport CCylonContextWrap
from pycylon.ctx.context cimport CylonContext
from pycylon.ctx.context import CylonContext

cdef api bint pyclon_is_context(object context)

cdef api CCylonContextWrap* pycylon_unwrap_context(object context)



