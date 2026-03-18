#pragma once

namespace math_precision {
	constexpr float epsilon = 5e-6f;

	__host__ __device__ bool nearly_equal(float a, float b) {
		float rel_tol = 1e-4f;
		return __builtin_fabs(a - b) <= __builtin_fmax(epsilon, rel_tol * __builtin_fmax(__builtin_fabs(a), __builtin_fabs(b)));
	}
}
