cmake_minimum_required(VERSION 3.26)

add_library(ray INTERFACE)
target_include_directories(ray INTERFACE include)

if (BUILD_TESTS)
	add_subdirectory(test)
endif()
