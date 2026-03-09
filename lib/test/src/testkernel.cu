#include "testkernel.h"
#include <cmath>
#include <curand_kernel.h>

__global__ void addNums(float* A, float* B, float* C, const int arrLength) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id >= arrLength) return;

	C[id] = A[id] + B[id];
}

__global__ void initArray(float* arr, const int arrLength, unsigned long seed) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id > arrLength) return;

	curandState state;
	curand_init(seed, id, 0, &state);

	arr[id] = curand_uniform(&state);
}


// Basic function which adds numbers via a cuda kernel
testResult launchTest() {
	const int arrLength = 4096;
	const int threads = 256;

	float* A = nullptr;
	float* B = nullptr;
	float* C = nullptr;

	cudaMallocManaged(&A, arrLength*sizeof(float));
    cudaMallocManaged(&B, arrLength*sizeof(float));
    cudaMallocManaged(&C, arrLength*sizeof(float));

	int blocks = ceil((float)arrLength / threads);

	long seed = 102746194;
	initArray<<<blocks, threads>>>(A, arrLength, seed);
	initArray<<<blocks, threads>>>(B, arrLength, seed << 27);
	addNums<<<blocks, threads>>>(A, B, C, arrLength);

    cudaDeviceSynchronize();

	testResult res;
	res.A = A;
	res.B = B;
	res.C = C;
	res.arrLength = arrLength;
	return res;
}
