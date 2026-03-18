#include "../test_runner.h"
#include "mat.cuh"
#include <cassert>

template<size_t r, size_t c>
__global__ void scale_mat_kernel(const mat<r,c>* m, const float* scalar, mat<r,c>* res) {
	*res = *scalar * *m;
}

template<size_t r, size_t c>
void scale_mat_cu() {
	mat<r,c> *m, *res;
	float *scalar;

	cudaMallocManaged(&m, sizeof(mat<r,c>));
	cudaMallocManaged(&res, sizeof(mat<r,c>));
    cudaMallocManaged(&scalar, sizeof(float));

	*m = init_mat<r,c>();
	*scalar = dist(rng);

	scale_mat_kernel<<<1,1>>>(m, scalar, res);
	cudaDeviceSynchronize();
	
	mat<r,c> check_mat;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			check_mat.data[i][j] = m->data[i][j] * *scalar;
		}
	}

	assert(check_mat == *res && *res == *scalar * *m);

	cudaFree(m);
	cudaFree(res);
	cudaFree(scalar);
}

template<size_t r, size_t c>
void scale_mat_cpp() {
	const mat<r,c> m = init_mat<r,c>();
	const float scalar = dist(rng);

	mat<r,c> res = m * scalar;

	mat<r,c> check_mat;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			check_mat.data[i][j] = m.data[i][j] * scalar;
		}
	}

	assert(check_mat == res && res == scalar * m);
}

template<size_t r1, size_t c1, size_t r2, size_t c2>
struct scale_mat {
	void operator()() {
		// Test for floating point accuracy on both CPU & GPU
		scale_mat_cpp<r1, c1>();
		scale_mat_cu<r1, c1>();
		
		// Hardcoded test for algorithm correctness
		scale_mat_example();
	}

	void scale_mat_example() {
        mat<3,3> m;
        constexpr float scalar = 2.0f;

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				m.data[i][j] = i + j;
			}
		}

		mat<3,3> res1 = m * scalar, res2 = scalar * m;

		assert(res1.data[0][0] == 0);
		assert(res1.data[0][1] == 2);
		assert(res1.data[0][2] == 4);
		assert(res1.data[1][0] == 2);
		assert(res1.data[1][1] == 4);
		assert(res1.data[1][2] == 6);
		assert(res1.data[2][0] == 4);
		assert(res1.data[2][1] == 6);
		assert(res1.data[2][2] == 8);
		assert(res1 == res2);
	}
};

int main() {
	run_tests<scale_mat, 2, 16, 2, 16, 2, 2, 2, 2>();
}

