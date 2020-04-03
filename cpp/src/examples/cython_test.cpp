//
// Created by vibhatha on 3/25/20.
//
#include "lib/Circle.h"
#include "iostream"

using namespace std;

int main(){


    shapes::Circle circle(0,0, 10);
    cout << "Circle Radius : " << circle.getRadius() << endl;
    cout << "Circle Circumference (Int) : " << circle.getCircumference() << endl;
    cout << "Circle Area (Int) : " << circle.getArea() << endl;





    return 0;
}