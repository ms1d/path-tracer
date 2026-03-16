#include "../test_runner.h"
#include "matrix.cuh"
#include <cassert>

template<size_t r, size_t c>
__global__ void scale_matrix_kernel(const matrix<r,c>* m, const float* scalar, matrix<r,c>* res) {
	*res = *scalar * *m;
}

template<size_t r, size_t c>
void scale_matrix_cu() {
	matrix<r,c> *m, *res;
	float *scalar;

	cudaMallocManaged(&m, sizeof(matrix<r,c>));
	cudaMallocManaged(&res, sizeof(matrix<r,c>));
    cudaMallocManaged(&scalar, sizeof(float));

	*m = init_matrix<r,c>();
	*scalar = dist(rng);

	scale_matrix_kernel<<<1,1>>>(m, scalar, res);
	cudaDeviceSynchronize();
	
	matrix<r,c> check_matrix;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			check_matrix.data[i][j] = m->data[i][j] * *scalar;
		}
	}

	assert(check_matrix == *res && *res == *scalar * *m);

	cudaFree(m);
	cudaFree(res);
	cudaFree(scalar);
}

template<size_t r, size_t c>
void scale_matrix_cpp() {
	const matrix<r,c> m = init_matrix<r,c>();
	const float scalar = dist(rng);

	matrix<r,c> res = m * scalar;

	matrix<r,c> check_matrix;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			check_matrix.data[i][j] = m.data[i][j] * scalar;
		}
	}

	assert(check_matrix == res && res == scalar * m);
}

template<size_t r1, size_t c1, size_t r2, size_t c2>
struct scale_matrix {
	void operator()() {
		// Test for floating point accuracy on both CPU & GPU
		scale_matrix_cpp<r1, c1>();
		scale_matrix_cu<r1, c1>();
		
		// Hardcoded test for algorithm correctness

	}
};

int main() {
	run_tests<scale_matrix, 2, 16, 2, 16, 2, 2, 2, 2>();
}

