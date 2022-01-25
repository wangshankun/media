#!/bin/sh
g++ -O3 -std=c++11 -o test cyto.cpp `pkg-config opencv --cflags --libs` 
