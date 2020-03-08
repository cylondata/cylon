//
// Created by vibhatha on 3/6/20.
//

#include "../lib/library.cpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pytwisterx, m) {
    py::class_<Request>(m, "Request")
            .def(py::init<const std::string &, int, std::string &>())
            .def("setName", &Request::setName)
            .def("getName", &Request::getName)
            .def("setMessage", &Request::setMessage)
            .def("getMessage", &Request::getMessage)
            .def("setBufSize", &Request::setBufSize)
            .def("getBufSize", &Request::getBufSize);
}