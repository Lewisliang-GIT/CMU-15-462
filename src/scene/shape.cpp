
#include "shape.h"
#include "../geometry/util.h"

namespace Shapes {

Vec2 Sphere::uv(Vec3 dir) {
	float u = std::atan2(dir.z, dir.x) / (2.0f * PI_F);
	if (u < 0.0f) u += 1.0f;
	float v = std::acos(-1.0f * std::clamp(dir.y, -1.0f, 1.0f)) / PI_F;
	return Vec2{u, v};
}

BBox Sphere::bbox() const {
	BBox box;
	box.enclose(Vec3(-radius));
	box.enclose(Vec3(radius));
	return box;
}

PT::Trace Sphere::hit(Ray ray) const {
	//A3T2
	// Sphere center at origin, radius = this->radius
	Vec3 oc = ray.point; // Ray origin relative to sphere center (origin)
	Vec3 dir = ray.dir;

	float a = dot(dir, dir);
	float b = 2.0f * dot(oc, dir);
	float c = dot(oc, oc) - radius * radius;

	float discriminant = b * b - 4.0f * a * c;

	PT::Trace ret;
	ret.origin = ray.point;
	ret.hit = false;
	ret.distance = 0.0f;
	ret.position = Vec3{};
	ret.normal = Vec3{};
	ret.uv = Vec2{};

	if (discriminant < 0.0f) return ret;

	float sqrt_disc = std::sqrt(discriminant);
	float t1 = (-b - sqrt_disc) / (2.0f * a);
	float t2 = (-b + sqrt_disc) / (2.0f * a);

	// Find the smallest t in [ray.dist_bounds.x, ray.dist_bounds.y]
	float t = std::numeric_limits<float>::max();
	if (t1 >= ray.dist_bounds.x && t1 <= ray.dist_bounds.y) t = t1;
	else if (t2 >= ray.dist_bounds.x && t2 <= ray.dist_bounds.y) t = t2;
	else return ret;

	ret.hit = true;
	ret.distance = t;
	ret.position = ray.point + t * ray.dir;
	ret.normal = (ret.position / radius).normalize(); // correct normal for sphere of radius
	ret.uv = uv(ret.normal);

	return ret;
}

Vec3 Sphere::sample(RNG &rng, Vec3 from) const {
	die("Sampling sphere area lights is not implemented yet.");
}

float Sphere::pdf(Ray ray, Mat4 pdf_T, Mat4 pdf_iT) const {
	die("Sampling sphere area lights is not implemented yet.");
}

Indexed_Mesh Sphere::to_mesh() const {
	return Util::closed_sphere_mesh(radius, 2);
}

} // namespace Shapes

bool operator!=(const Shapes::Sphere& a, const Shapes::Sphere& b) {
	return a.radius != b.radius;
}

bool operator!=(const Shape& a, const Shape& b) {
	if (a.shape.index() != b.shape.index()) return false;
	return std::visit(
		[&](const auto& shape) {
			return shape != std::get<std::decay_t<decltype(shape)>>(b.shape);
		},
		a.shape);
}
