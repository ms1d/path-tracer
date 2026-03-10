#include "ray.h"
#include "vec3.h"

Ray::Ray(const vec3& origin, const vec3& direction) : origin(origin), direction(direction) {}

vec3 Ray::GetPoint(float t) const {
    return origin + t * direction;
}
