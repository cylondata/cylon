//
// Created by vibhatha on 3/25/20.
//
#include "lib/Circle.h"
#include "iostream"
#include "data/table_builder.h"
#include "status.cpp"

using namespace std;

int main(){


    shapes::Circle circle(0,0, 10);
    cout << "Circle Radius : " << circle.getRadius() << endl;
    cout << "Circle Circumference (Int) : " << circle.getCircumference() << endl;
    cout << "Circle Area (Int) : " << circle.getArea() << endl;


    twisterx::Status  s = twisterx::data::read_csv("s", "s");

    cout << s.get_code() << endl;



    return 0;
}