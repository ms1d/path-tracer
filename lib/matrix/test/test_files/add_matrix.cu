#include "../test_runner.h"
#include "matrix.cuh"
#include <cassert>

template<size_t r, size_t c>
__global__ void add_matrix_kernel(const matrix<r,c> *m1, const matrix<r,c> *m2, matrix<r,c> *res) {
	*res = *m1 + *m2;
}

template<size_t r, size_t c>
void add_matrix_cu() {
	matrix<r,c> *m1, *m2, *res;

	cudaMallocManaged(&m1, sizeof(matrix<r,c>));
	cudaMallocManaged(&m2, sizeof(matrix<r,c>));
	cudaMallocManaged(&res, sizeof(matrix<r,c>));

	*m1 = init_matrix<r,c>();
    *m2 = init_matrix<r,c>();

	add_matrix_kernel<<<1,1>>>(m1, m2, res);
	cudaDeviceSynchronize();

	assert(*res == *m1 + *m2);

	cudaFree(m1);
    cudaFree(m2);
	cudaFree(res);
}

template<size_t r, size_t c>
void add_matrix_cpp() {
	const matrix<r,c> m1 = init_matrix<r,c>(), m2 = init_matrix<r,c>();
	matrix<r,c> res = m1 + m2;

	assert(res == m1 + m2);
}

template<size_t r1, size_t c1, size_t r2, size_t c2>
struct add_matrix {
	void operator()() {
		add_matrix_cpp<r1, c1>();
		add_matrix_cu<r1, c1>();
	}
};

int main() {
	run_tests<add_matrix, 2, 16, 2, 16, 2, 2, 2, 2>();
}
