#pragma once

namespace math_precision {
	constexpr float epsilon = 1e-6f;

	__host__ __device__ constexpr bool nearly_equal(float a, float b) {
		constexpr float rel_tol = 2e-6f;
		return __builtin_fabs(a - b) <= __builtin_fmax(epsilon, rel_tol * __builtin_fmax(__builtin_fabs(a), __builtin_fabs(b)));
	}
}
