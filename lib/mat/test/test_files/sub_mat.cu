#include "../test_runner.h"
#include "mat.cuh"
#include <cassert>

template<size_t r, size_t c>
__global__ void sub_mat_kernel(const mat<r,c> *m1, const mat<r,c> *m2, mat<r,c> *res) {
	*res = *m1 - *m2;
}

template<size_t r, size_t c>
void sub_mat_cu() {
	mat<r,c> *m1, *m2, *res;

	cudaMallocManaged(&m1, sizeof(mat<r,c>));
	cudaMallocManaged(&m2, sizeof(mat<r,c>));
	cudaMallocManaged(&res, sizeof(mat<r,c>));

	*m1 = init_mat<r,c>();
    *m2 = init_mat<r,c>();

	sub_mat_kernel<<<1,1>>>(m1, m2, res);
	cudaDeviceSynchronize();

	assert(*res == *m1 - *m2);

	cudaFree(m1);
    cudaFree(m2);
	cudaFree(res);
}

template<size_t r, size_t c>
void sub_mat_cpp() {
	const mat<r,c> m1 = init_mat<r,c>(), m2 = init_mat<r,c>();
	mat<r,c> res = m1 - m2;

	assert(res == m1 - m2);
}

template<size_t r1, size_t c1, size_t r2, size_t c2>
struct sub_mat {
	void operator()() {
		// Test for floating point accuracy on both CPU & GPU
		sub_mat_cpp<r1, c1>();
		sub_mat_cu<r1, c1>();

		// Hardcoded test for algorithm correctness
		sub_mat_example();
	}

	void sub_mat_example() {
		mat<3,3> m1, m2;

		for (size_t i = 0; i < 3; i++) {
			for (size_t j = 0; j < 3; j++) {
				m1.data[i][j] = i + j;
				m2.data[i][j] = i * j;
			}
		}

		mat<3,3> res1 = m1 - m2, res2 = m2 - m1;

		assert(res1 == -1 * res2);
		assert(res1.data[0][0] == 0);
		assert(res1.data[0][1] == 1);
		assert(res1.data[0][2] == 2);
		assert(res1.data[1][0] == 1);
		assert(res1.data[1][1] == 1);
		assert(res1.data[1][2] == 1);
		assert(res1.data[2][0] == 2);
		assert(res1.data[2][1] == 1);
		assert(res1.data[2][2] == 0);
	}
};

int main() {
	run_tests<sub_mat, 2, 16, 2, 16, 2, 2, 2, 2>();
}

