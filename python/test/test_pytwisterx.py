from pytwisterx.geometry import PyCircle

x0=0
y0=0
radius=10
_exp_circumference = 2 * 3 * 10
_exp_area = 3 * 10 * 10
circle = PyCircle(0, 0, 10)

assert circle.getRadius() == radius
assert circle.getCircumference() == _exp_circumference
assert circle.getArea() == _exp_area

#print("Radius {}, Int Circumference {}, Int Area {}".format(circle.getRadius(), circle.getCircumference(), circle.getArea()))
