#pragma once

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include "vec.h"
#include <iostream>

struct Ray {
	public:
		vec<3> origin;
        vec<3> direction;

		__host__ __device__ Ray(const vec<3>& origin, const vec<3>& direction);

		__host__ __device__ vec<3> GetPoint(float t) const;

		friend std::ostream& operator<<(std::ostream& os, const Ray& v);
};
