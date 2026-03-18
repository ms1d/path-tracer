#include "../test_runner.h"
#include "mat.cuh"
#include <cassert>

template<size_t r, size_t c>
__global__ void det_mat_kernel(const mat<r,c> *m, float *det) {
	*det = m->det();
}

template<size_t r, size_t c>
void det_mat_cu() {
	mat<r,c> *m;
	float *det;

	cudaMallocManaged(&m, sizeof(mat<r,c>));
	cudaMallocManaged(&det, sizeof(float));

	*m = init_mat<r,c>();

	det_mat_kernel<<<1,1>>>(m, det);
	cudaDeviceSynchronize();

	assert(__builtin_fabsf(*det - m->det()) < epsilon);

	cudaFree(m);
	cudaFree(det);
}

template<size_t r, size_t c>
void det_mat_cpp() {
    const mat<r,c> m = init_mat<r,c>();
	float det = m.det();

	assert(__builtin_fabsf(det - m.det()) < epsilon);
}

template<size_t r1, size_t c1, size_t r2, size_t c2>
struct det_mat {
	// Only need to test valid sizes
	void operator()() requires(r1 != c1) {}

	void operator()() requires(r1 == c1) {
		// Test for floating point accuracy on both CPU & GPU
		det_mat_cpp<r1,c1>();
		det_mat_cu<r1,c1>();

		// Hardcoded test for algorithm correctness
		det_mat_example();
	}

	void det_mat_example() {
		mat<3,3> m;

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				m.data[i][j] = i + j;
			}
		}

		assert(m.det() == 0);
	}
};

int main() {
	run_tests<det_mat, 2, 16, 2, 16, 2, 2, 2, 2>();
}
