#pragma once

#include "vec.cuh"
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

template <size_t r, size_t c>
struct matrix {



	float data[r][c];



	__host__ __device__ matrix() {}



	__host__ __device__ matrix& operator+=(const matrix& other) {
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				data[i][j] += other.data[i][j];
			}
		}

		return *this;
	}

	__host__ __device__ matrix operator+(const matrix& other) const {
		matrix res = *this;
		res += other;
		return res;
	}



	__host__ __device__ matrix& operator-=(const matrix& other) {
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				data[i][j] -= other.data[i][j];
			}
		}

		return *this;
	}

	__host__ __device__ matrix operator-(const matrix& other) const {
		matrix res = *this;
		res -= other;
		return res;
	}



	__host__ __device__ matrix<r-1, c-1> get_minor(int row, int col) const {
		matrix<r-1,c-1> res;
		int curr_row, curr_col = 0;

		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				if (i == row || j == col) continue;

				res[curr_row][curr_col] = data[i][j];

				curr_col++;
			}

			curr_row++;
		}

		return res;
	}

	__host__ __device__ float det() const {
		static_assert(r == c, "Matrix must be square");

		// Base case for recursion (1 by 1 matrix)
		if (r == 1) return data[0][0];

		float det = 0;

		int sign = 1;
		for (int j = 0; j < c; j++) {
			det += sign * data[0][j] * get_minor(0, j);
			sign *= -1;
		}

		return det;
	};



	__host__ __device__ matrix& operator*=(float scalar) {
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				data[i][j] *= scalar;
			}
		}

		return *this;
	}

	__host__ __device__ matrix operator*(float scalar) const {
		matrix m = *this;
		m *= scalar;
		return m;
	}



	__host__ __device__ vec<r> operator*(const vec<c>& v) const {
		vec<r> res;

		for (size_t i = 0; i < r; ++i) {
			float sum = 0;
			for (size_t j = 0; j < c; ++j)
				sum += data[i][j] * v.data[j];
			res.data[i] = sum;
		}

		return res;
	}


	__host__ __device__ bool operator==(const matrix& other) const {
		constexpr float epsilon = 2e-6;

		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				if (fabs(data[i][j] - other.data[i][j]) > epsilon) return false;
			}
		}

		return true;
	}



};

template <size_t r, size_t c>
__host__ __device__ matrix<r, c> operator*(float scalar, const matrix<r, c>& m) {
	return m * scalar;
}

template<size_t r1, size_t r2, size_t c1, size_t c2>
__host__ __device__ matrix<r1, c2> operator*(const matrix<r1, c1>& m1, const matrix<r2, c2>& m2) {
	static_assert(c1 == r2, "Matrix dimensions must match");

	matrix<r1,c2> res;

	for (size_t i = 0; i < r1; ++i) {
		for (size_t j = 0; j < c2; ++j) {
			float sum = 0;

			for (size_t k = 0; k < c1; ++k)
				sum += m1.data[i][k] * m2.data[k][j];

			res.data[i][j] = sum;
		}
	}

	return res;
}
