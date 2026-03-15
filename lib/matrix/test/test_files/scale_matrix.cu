#include "../test_runner.h"
#include "matrix.cuh"
#include <cassert>

template<size_t r, size_t c>
__global__ void scale_matrix_kernel() {

}

template<size_t r, size_t c>
void scale_matrix_cu() {

}

template<size_t r, size_t c>
void scale_matrix_cpp() {
	const matrix<r,c> m = init_matrix<r,c>();
	const float scalar = dist(rng);

	matrix<r,c> res = m * scalar;

	// Assert please
}

template<size_t r1, size_t c1, size_t r2, size_t c2>
struct scale_matrix {
	void operator()() {
		scale_matrix_cpp<r1, c1>();
		scale_matrix_cu<r1, c1>();
	}
};

int main() {
	run_tests<scale_matrix, 2, 16, 2, 16, 2, 16, 2, 16>();
}

