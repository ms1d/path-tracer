cmake_minimum_required(VERSION 3.26)

add_library(matrix INTERFACE)
target_include_directories(matrix INTERFACE include)
target_link_libraries(matrix INTERFACE vec)

if (BUILD_TESTS)
    add_subdirectory(test)
endif()
