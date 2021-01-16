//
//  main.cpp
//  HelloWorld
//
//  Created by Eric Tang on 6/5/20.
//  Copyright © 2020 Eric Tang. All rights reserved.
//


#include <iostream>
 
void doNothing(int&) // Don't worry about what & is for now, we're just using it to trick the compiler into thinking variable x is used
{
}
 
int main()
{
    // define an integer variable named x
    int x; // this variable is uninitialized
 
    doNothing(x); // make the compiler think we're assigning a value to this variable
 
    // print the value of x to the screen (who knows what we'll get, because x is uninitialized)
    std::cout << x << '\n';
 
    return 0;
}
