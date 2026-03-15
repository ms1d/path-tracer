#include "../test_runner.h"
#include "matrix.cuh"
#include <cassert>

template<size_t r1, size_t c1, size_t r2, size_t c2>
__global__ void mult_matrix_kernel(const matrix<r1,c1>* m1, const matrix<r2,c2>* m2, matrix<r1,c2>* res) {
	if constexpr (c1 != r2) return;
	*res = *m1 * *m2;
}

template<size_t r1, size_t c1, size_t r2, size_t c2>
void mult_matrix_cu() {
	if constexpr (c1 != r2) return;
	matrix<r1,c1> *m1;
	matrix<r2,c2> *m2;
	matrix<r1,c2> *res;

	cudaMallocManaged(&m1, sizeof(matrix<r1,c1>));
	cudaMallocManaged(&m2, sizeof(matrix<r2,c2>));
	cudaMallocManaged(&res, sizeof(matrix<r1,c2>));

	*m1 = init_matrix<r1,c1>();
    *m2 = init_matrix<r2,c2>();

	mult_matrix_kernel<<<1,1>>>(m1, m2, res);
	cudaDeviceSynchronize();

	matrix<r1,c2> check_matrix;

	for (int i = 0; i < r1; i++) {
		for (int j = 0; j < c2; j++) {
			float sum = 0;

			for (int k = 0; k < c1; k++) {
				sum += m1->data[i][k] * m2->data[k][j];
			}

			check_matrix.data[i][j] = sum;
		}
	}

	assert(check_matrix == *res);
	
	cudaFree(m1);
	cudaFree(m2);
	cudaFree(res);
}

template<size_t r1, size_t c1, size_t r2, size_t c2>
void mult_matrix_cpp() {
	if constexpr (c1 != r2) return;

	const matrix<r1,c1> m1 = init_matrix<r1,c1>();
	const matrix<r2,c2> m2 = init_matrix<r2,c2>();

	matrix<r1,c2> res = m1 * m2, check_matrix;

	for (int i = 0; i < r1; i++) {
		for (int j = 0; j < c2; j++) {
			float sum = 0;

			for (int k = 0; k < c1; k++) {
				sum += m1.data[i][k] * m2.data[k][j];
			}

			check_matrix.data[i][j] = sum;
		}
	}

	assert(check_matrix == res);
}

template<size_t r1, size_t c1, size_t r2, size_t c2>
struct mult_matrix {
	void operator()() {
		if constexpr (c1 == r2) {
			mult_matrix_cpp<r1, c1, r2, c2>();
			mult_matrix_cu<r1, c1, r2, c2>();
		}
	}
};

int main() {
	run_tests<mult_matrix, 2, 16, 2, 16, 2, 16, 2, 16>();
}

