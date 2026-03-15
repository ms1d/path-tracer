#include "ray.cuh"
#include "vec.cuh"

__host__ __device__ Ray::Ray(const vec<3>& origin, const vec<3>& direction) : origin(origin), direction(direction) {}

__host__ __device__ vec<3> Ray::GetPoint(float t) const {
    return origin + t * direction;
}
