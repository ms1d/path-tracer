#include "../test_runner.h"
#include "matrix.cuh"
#include "vec.cuh"
#include <cassert>

template<size_t r, size_t c>
__global__ void transform_vector_kernel() {

}

template<size_t r, size_t c>
void transform_vector_cu() {

}

template<size_t r, size_t c>
void transform_vector_cpp() {

}

template<size_t r1, size_t c1, size_t r2, size_t c2>
struct transform_vector {
	void operator()() {
		transform_vector_cpp<r1, c1>();
		transform_vector_cu<r1, c1>();
	}
};

int main() {
	run_tests<transform_vector, 2, 16, 2, 16, 2, 16, 2, 16>();
}

