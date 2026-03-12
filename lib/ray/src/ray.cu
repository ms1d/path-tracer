#include "ray.h"
#include "vec.h"

__host__ __device__ Ray::Ray(const vec<3>& origin, const vec<3>& direction) : origin(origin), direction(direction) {}

__host__ __device__ vec<3> Ray::GetPoint(float t) const {
    return origin + t * direction;
}
