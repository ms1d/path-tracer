#pragma once

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include "vec3.h"
#include <iostream>

struct Ray {
	public:
		vec3 origin;
        vec3 direction;

		__host__ __device__ Ray(const vec3& origin, const vec3& direction);

		__host__ __device__ vec3 GetPoint(float t) const;

		friend std::ostream& operator<<(std::ostream& os, const Ray& v);
};
