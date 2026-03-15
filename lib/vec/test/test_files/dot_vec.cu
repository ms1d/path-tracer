#include "vec.cuh"
#include "../test_runner.h"
#include <cassert>
#include <cstdio>

template<size_t dim>
__global__ void dot_vec_kernel(const vec<dim>* v1, const vec<dim>* v2, float* dot) {
	*dot = *v1 * *v2;
}

template<size_t dim>
void dot_vec_cu() {
	const float epsilon = 2e-6;

	vec<dim> *v1, *v2;
	float *dot;

	cudaMallocManaged(&v1, sizeof(vec<dim>));
    cudaMallocManaged(&v2, sizeof(vec<dim>));
    cudaMallocManaged(&dot, sizeof(float));

	*v1 = init_vec<dim>();
	*v2 = init_vec<dim>();

	dot_vec_kernel<<<1, 1>>>(v1, v2, dot);
	cudaDeviceSynchronize();

	float sum = 0;
	for (size_t i = 0; i < dim; i++)
		sum += v1->data[i] * v2->data[i];
	
	assert(sum - epsilon < *dot && sum + epsilon > *dot);

	cudaFree(v1);
	cudaFree(v2);
	cudaFree(dot);
}

template<size_t dim>
void dot_vec_cpp() {
	const vec<dim> v1 = init_vec<dim>(), v2 = init_vec<dim>();

	float dot = v1 * v2;

	float sum = 0;
	for (size_t i = 0; i < dim; i++)
		sum += v1.data[i] * v2.data[i];

	assert(sum == dot);
}

template<size_t dim>
struct dot_vec {
	void operator()() {
		dot_vec_cpp<dim>();
		dot_vec_cu<dim>();
	}
};

int main() {
	run_tests<dot_vec,2,64>();
}
