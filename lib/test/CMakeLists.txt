cmake_minimum_required(VERSION 3.26)

add_library(test src/test.cpp)
target_include_directories(test PUBLIC include)
