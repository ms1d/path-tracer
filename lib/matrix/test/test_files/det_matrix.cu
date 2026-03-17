#include "../test_runner.h"
#include "matrix.cuh"
#include <cassert>

template<size_t r, size_t c>
__global__ void det_matrix_kernel(const matrix<r,c> *m, float *det) {
	*det = m->det();
}

template<size_t r, size_t c>
void det_matrix_cu() {
	matrix<r,c> *m;
	float *det;

	cudaMallocManaged(&m, sizeof(matrix<r,c>));
	cudaMallocManaged(&det, sizeof(float));

	*m = init_matrix<r,c>();

	det_matrix_kernel<<<1,1>>>(m, det);
	cudaDeviceSynchronize();

	assert(fabs(*det - m->det()) < epsilon);

	cudaFree(m);
	cudaFree(det);
}

template<size_t r, size_t c>
void det_matrix_cpp() {
    const matrix<r,c> m = init_matrix<r,c>();
	float det = m.det();

	assert(fabs(det - m.det()) < epsilon);
}

template<size_t r1, size_t c1, size_t r2, size_t c2>
struct det_matrix {
	// Only need to test valid sizes
	void operator()() requires(r1 != c1) {}

	void operator()() requires(r1 == c1) {
		// Test for floating point accuracy on both CPU & GPU
		det_matrix_cpp<r1,c1>();
		det_matrix_cu<r1,c1>();

		// Hardcoded test for algorithm correctness
		det_matrix_example();
	}

	void det_matrix_example() {
		matrix<3,3> m;

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				m.data[i][j] = i + j;
			}
		}

		assert(m.det() == 0);
	}
};

int main() {
	run_tests<det_matrix, 2, 16, 2, 16, 2, 2, 2, 2>();
}
