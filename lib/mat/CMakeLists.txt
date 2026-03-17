cmake_minimum_required(VERSION 3.26)

add_library(mat INTERFACE)
target_include_directories(mat INTERFACE include)
target_link_libraries(mat INTERFACE vec)

if (BUILD_TESTS)
    add_subdirectory(test)
endif()
