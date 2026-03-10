#pragma once
#define RAY_STRUCT_H

#include "vec3.h"

struct Ray {
	public:
		vec3 origin;
        vec3 direction;

		Ray(const vec3& origin, const vec3& direction);

		vec3 GetPoint(float t) const;
};
