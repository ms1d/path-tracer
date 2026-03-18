cmake_minimum_required(VERSION 3.26)

add_library(vec INTERFACE)
target_include_directories(vec INTERFACE include)
target_link_libraries(vec INTERFACE common)

if (BUILD_TESTS)
	add_subdirectory(test)
endif()
