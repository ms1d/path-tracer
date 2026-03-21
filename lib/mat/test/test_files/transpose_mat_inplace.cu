#include "../test_runner.h"
#include <cassert>
#include "mat.cuh"

template<size_t dim>
__global__ void transpose_mat_inplace_kernel(mat<dim,dim> *m) {
	m->transpose_inplace();
}

template<size_t dim>
void transpose_mat_inplace_cu() {
	mat<dim,dim> *m, mcpy;

	cudaMallocManaged(&m, sizeof(mat<dim,dim>));

	*m = init_mat<dim,dim>();
	mcpy = *m;

	transpose_mat_inplace_kernel<<<1,1>>>(m);
	cudaDeviceSynchronize();
	
	for (size_t i = 0; i < dim; i++) {
		for (size_t j = 0; j < dim; j++) {
			std::cout << m->data[i][j] << " vs " << mcpy.data[j][i] << std::endl;
			assert(m->data[i][j] == mcpy.data[j][i]);
		}
	}

	cudaFree(m);
}



template<size_t dim>
void transpose_mat_inplace_cpp() {
	mat<dim,dim> m = init_mat<dim,dim>(), mcpy = m;
	m.transpose_inplace();

	for (size_t i = 0; i < dim; i++) {
		for (size_t j = 0; j < dim; j++) {
			assert(m.data[i][j] == mcpy.data[j][i]);
		}
	}
}


template<size_t r1, size_t c1, size_t r2, size_t c2>
struct transpose_mat_inplace {
	void operator()() {
		if constexpr (r1 == c1) {
			// Test for floating point accuracy on both CPU & GPU
			transpose_mat_inplace_cpp<r1>();
			transpose_mat_inplace_cu<r1>();

			// Hardcoded test for algorithm correctness
			transpose_mat_inplace_example();
		}
	}

	void transpose_mat_inplace_example() {

	}
};

int main() {
	run_tests<transpose_mat_inplace, 2, 16, 2, 16, 2, 2, 2, 2>();
}
