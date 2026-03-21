#include "../test_runner.h"
#include "precision.cuh"
#include "vec.cuh"
#include <cassert>



template<size_t dim>
__global__ void norm_vec_kernel(const vec<dim> *v1, vec<dim> *v2) {
	*v2 = v1->norm();
}

template<size_t dim>
void norm_vec_cu() {
	vec<dim> *v1, *v2;

	cudaMallocManaged(&v1, sizeof(vec<dim>));
    cudaMallocManaged(&v2, sizeof(vec<dim>));

	*v1 = init_vec<dim>();

	norm_vec_kernel<<<1, 1>>>(v1, v2);
	cudaDeviceSynchronize();

	// Obviously magnitude should be 1
	assert(math_precision::nearly_equal(v2->mag(), 1.0f));

	// Obviously vectors should point in the same direction
	assert(math_precision::nearly_equal(*v2 * *v1, v1->mag()));

	cudaFree(v1);
	cudaFree(v2);
}



template<size_t dim>
void norm_vec_cpp() {
	vec<dim> v1 = init_vec<dim>(), v2 = v1.norm();

	// Obviously magnitude should be 1
	assert(math_precision::nearly_equal(v2.mag(), 1.0f));

	// Obviously vectors should point in the same direction
	assert(math_precision::nearly_equal(v2 * v1, v1.mag()));
}



template<size_t dim>
struct norm_vec {
	void operator()() {
		// Test for floating point accuracy on both CPU & GPU
		norm_vec_cpp<dim>();
		norm_vec_cu<dim>();

		// Hardcoded test for algorithm correctness
		norm_vec_example();
	}

	void norm_vec_example() {
		vec<3> v1{1,2,3}, v2 = v1.norm();
		assert(math_precision::nearly_equal(v2.mag(), 1.0f));
		assert(v2 == vec<3>(0.267261,0.534522,0.801784));
	}
};



int main() {
	run_tests<norm_vec, 2, 64>();
}
