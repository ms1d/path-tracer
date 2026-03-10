#include "ray.h"
#include <iostream>

std::ostream& operator<<(std::ostream& os, const Ray& r) {
    return os << "(" << r.origin << ", " << r.direction << ")";
}
