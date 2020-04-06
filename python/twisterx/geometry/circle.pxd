cdef extern from "../../../cpp/src/twisterx/lib/Circle.h" namespace "shapes":
    cdef cppclass Circle:
        Circle(int, int, int)
        int x0, y0, radius
        int getRadius()
        int getCircumference()
        int getArea()