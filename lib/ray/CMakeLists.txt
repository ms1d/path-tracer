cmake_minimum_required(VERSION 3.26)

add_library(ray INTERFACE)
target_include_directories(ray INTERFACE include)
target_link_libraries(ray INTERFACE common vec)

if (BUILD_TESTS)
	add_subdirectory(test)
endif()
