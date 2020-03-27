from twisterx.geometry.circle cimport Circle

cdef class PyCircle:

    cdef Circle *thisptr

    def __cinit__(self, int x0, int y0, int radius):
        self.thisptr = new Circle(x0, y0, radius) #new CCircle.CCircle(x0, y0, radius)

    def __dealloc__(self):
        del self.thisptr

    cpdef getCircumference(self):
        return self.thisptr.getCircumference()

    cpdef getRadius(self):
        return self.thisptr.getRadius()

    cpdef getArea(self):
        return self.thisptr.getArea()
