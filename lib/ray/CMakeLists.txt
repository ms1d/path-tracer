cmake_minimum_required(VERSION 3.26)

add_library(ray src/ray.cpp)
target_include_directories(ray PUBLIC include)
target_link_libraries(ray PUBLIC vec3)

set_target_properties(ray  PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)
