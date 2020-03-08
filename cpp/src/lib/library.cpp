//
// Created by vibhatha on 3/6/20.
//

#include "library.h"
#include <iostream>


struct Request {
    Request(const std::string &name, int bufSize, std::string &message) : name(name), bufSize(bufSize),
                                                                          message(message) {}

    void setName(const std::string &name_) { name = name_; }

    const std::string &getName() const { return name; }

    void setMessage(std::string &message_) { message = message_; }

    std::string &getMessage() { return message; }

    void setBufSize(int size_) { bufSize = size_; }

    int getBufSize() { return bufSize; }

    std::string name;
    int bufSize;
    std::string message;
};
