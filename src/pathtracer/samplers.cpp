
#include "samplers.h"
#include "../util/rand.h"

constexpr bool IMPORTANCE_SAMPLING = true;

namespace Samplers {

Vec2 Rect::sample(RNG &rng) const {
	//A3T1 - step 2 - supersampling

	// Return a point selected uniformly at random from the rectangle [0,size.x)x[0,size.y)
	// Useful function: rng.unit()

	float x = rng.unit() * size.x;
	float y = rng.unit() * size.y;
	return Vec2{x, y};
}

float Rect::pdf(Vec2 at) const {
	if (at.x < 0.0f || at.x > size.x || at.y < 0.0f || at.y > size.y) return 0.0f;
	return 1.0f / (size.x * size.y);
}

Vec2 Circle::sample(RNG &rng) const {
	//A3EC - bokeh - circle sampling

    // Return a point selected uniformly at random from a circle defined by its
	// center and radius.
    // Useful function: rng.unit()

    return Vec2{};
}

float Circle::pdf(Vec2 at) const {
	//A3EC - bokeh - circle pdf

	// Return the pdf of sampling the point 'at' for a circle defined by its
	// center and radius.

    return 1.f;
}

Vec3 Point::sample(RNG &rng) const {
	return point;
}

float Point::pdf(Vec3 at) const {
	return at == point ? 1.0f : 0.0f;
}

Vec3 Triangle::sample(RNG &rng) const {
	float u = std::sqrt(rng.unit());
	float v = rng.unit();
	float a = u * (1.0f - v);
	float b = u * v;
	return a * v0 + b * v1 + (1.0f - a - b) * v2;
}

float Triangle::pdf(Vec3 at) const {
	float a = 0.5f * cross(v1 - v0, v2 - v0).norm();
	float u = 0.5f * cross(at - v1, at - v2).norm() / a;
	float v = 0.5f * cross(at - v2, at - v0).norm() / a;
	float w = 1.0f - u - v;
	if (u < 0.0f || v < 0.0f || w < 0.0f) return 0.0f;
	if (u > 1.0f || v > 1.0f || w > 1.0f) return 0.0f;
	return 1.0f / a;
}

Vec3 Hemisphere::Uniform::sample(RNG &rng) const {

	float Xi1 = rng.unit();
	float Xi2 = rng.unit();

	float theta = std::acos(Xi1);
	float phi = 2.0f * PI_F * Xi2;

	float xs = std::sin(theta) * std::cos(phi);
	float ys = std::cos(theta);
	float zs = std::sin(theta) * std::sin(phi);

	return Vec3(xs, ys, zs);
}

float Hemisphere::Uniform::pdf(Vec3 dir) const {
	if (dir.y < 0.0f) return 0.0f;
	return 1.0f / (2.0f * PI_F);
}

Vec3 Hemisphere::Cosine::sample(RNG &rng) const {

	float phi = rng.unit() * 2.0f * PI_F;
	float cos_t = std::sqrt(rng.unit());

	float sin_t = std::sqrt(1 - cos_t * cos_t);
	float x = std::cos(phi) * sin_t;
	float z = std::sin(phi) * sin_t;
	float y = cos_t;

	return Vec3(x, y, z);
}

float Hemisphere::Cosine::pdf(Vec3 dir) const {
	if (dir.y < 0.0f) return 0.0f;
	return dir.y / PI_F;
}

Vec3 Sphere::Uniform::sample(RNG &rng) const {
	//A3T7 - sphere sampler

    // Generate a uniformly random point on the unit sphere.
    // Tip: start with Hemisphere::Uniform

    float z = 2.0f * rng.unit() - 1.0f; // cos(theta) in [-1,1]
    float phi = 2.0f * PI_F * rng.unit();
    float r = std::sqrt(1.0f - z * z);
    float x = r * std::cos(phi);
    float y = r * std::sin(phi);
    return Vec3(x, z, y); // y is up in Scotty3D, so z is vertical
}

float Sphere::Uniform::pdf(Vec3 dir) const {
	return 1.0f / (4.0f * PI_F);
}

Sphere::Image::Image(const HDR_Image& image) {
    //A3T7 - image sampler init

    // Set up importance sampling data structures for a spherical environment map image.
    const auto [_w, _h] = image.dimension();
    w = _w;
    h = _h;
    _pdf.clear();
    _cdf.clear();
    float total = 0.0f;
    // Compute PDF and CDF for each pixel
    for (int y = 0; y < h; ++y) {
        float theta = PI_F * (float(y) + 0.5f) / float(h); // [0, pi]
        float sin_theta = std::sin(theta);
        for (int x = 0; x < w; ++x) {
            Spectrum c = image.at(x, y);
            float luminance = c.luma();
            float p = luminance * sin_theta;
            _pdf.push_back(p);
            total += p;
        }
    }
    // Normalize PDF and build CDF
    float accum = 0.0f;
    for (size_t i = 0; i < _pdf.size(); ++i) {
        _pdf[i] /= total;
        accum += _pdf[i];
        _cdf.push_back(accum);
    }
}

Vec3 Sphere::Image::sample(RNG &rng) const {
	if(!IMPORTANCE_SAMPLING) {
		// Step 1: Uniform sampling
		Uniform uniform_sampler;
		return uniform_sampler.sample(rng);
	} else {
		// Step 2: Importance sampling
		float p = rng.unit();
		size_t i = std::upper_bound(_cdf.begin(), _cdf.end(), p) - _cdf.begin();
		size_t row_idx = i / w;
		size_t col_idx = i % w;
		float theta = PI_F - ((float)row_idx + 0.5f) / (float)h * PI_F;
		float phi = ((float)col_idx + 0.5f) / (float)w * 2.0f * PI_F;
		float y_s = std::cos(theta);
		float x_s = std::sin(theta) * std::cos(phi);
		float z_s = std::sin(theta) * std::sin(phi);
		return Vec3{x_s, y_s, z_s};
	}
}

float Sphere::Image::pdf(Vec3 dir) const {
	if(!IMPORTANCE_SAMPLING) {
		// Step 1: Uniform sampling
		Uniform uniform_sampler;
		return uniform_sampler.pdf(dir);
	} else {
		// Importance sampling PDF
		 float theta, phin;
        theta = acos(dir.z);
        phin = atan2(dir.y, dir.z);
        if (phin < 0.0f) phin += 2*PI_F;

        float u = phin / (2 * PI_F) * w;
        float v = (1.0f - theta / PI_F) * h;

        int x = static_cast<int>(u) % w;
        int y = static_cast<int>(v) % h;
        int idx = y * w + x;

    	return (_cdf[idx])/(2*PI_F*PI_F*sin(theta));
	}
}

} // namespace Samplers
