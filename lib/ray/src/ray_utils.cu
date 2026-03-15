#include "ray.cuh"
#include "vec3.cuh"

// Moller-Trumbore ray-triangle intersection
__host__ __device__ vec3 MTRayTriIntersect(Ray ray, vec3 tri) {

}

// Get the t value of the intersection of a ray and a point if it exists
// Negative values of t indicate no intersection
__host__ __device__ float getIntersectionT(Ray ray, vec3 point) {
	vec3 u = (point - ray.origin);
	
	float t = ray.direction.x != 0 ? __fdividef(u.x, ray.direction.x) : -1;
	float y = ray.origin.y + t * ray.direction.y;
	float z = ray.origin.z + t * ray.direction.z;

	if (y == point.y && z == point.z) return t;
	return -1;
}
