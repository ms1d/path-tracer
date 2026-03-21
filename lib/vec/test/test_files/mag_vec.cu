#include "vec.cuh"
#include "../test_runner.h"
#include <cassert>



template<size_t dim>
__global__ void mag_vec_kernel(const vec<dim> *v1, float *mag) {
	*mag = v1->mag();
}

template<size_t dim>
void mag_vec_cu() {
	vec<dim> *v1;
	float *mag;

	cudaMallocManaged(&v1, sizeof(vec<dim>));
	cudaMallocManaged(&mag, sizeof(float));

	*v1 = init_vec<dim>();

	mag_vec_kernel<<<1, 1>>>(v1, mag);
	cudaDeviceSynchronize();

	float sum = 0;
	for (size_t i = 0; i < dim; i++) sum += v1->data[i] * v1->data[i];
	sum = __builtin_sqrtf(sum);

	assert(math_precision::nearly_equal(sum, *mag));

	cudaFree(v1);
	cudaFree(mag);
}



template<size_t dim>
void mag_vec_cpp() {
	const vec<dim> v1 = init_vec<dim>();
	float mag = v1.mag();

	float sum = 0;
	for (size_t i = 0; i < dim; i++) sum += v1.data[i] * v1.data[i];
	sum = __builtin_sqrtf(sum);

	assert(math_precision::nearly_equal(sum, mag));
}



template<size_t dim>
struct mag_vec {
	void operator()() {
		// Test for floating point accuracy on both CPU & GPU
		mag_vec_cpp<dim>();
		mag_vec_cu<dim>();

		// Hardcoded test for algorithm correctness
		mag_vec_example();
	}

	void mag_vec_example() {
		vec<3> v{1,2,3};
		assert(math_precision::nearly_equal(v.mag(), __builtin_sqrtf(14)));
	}
};



int main() {
	run_tests<mag_vec, 2, 64>();
}

