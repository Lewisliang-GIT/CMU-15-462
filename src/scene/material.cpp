
#include "material.h"
#include "../util/rand.h"

namespace Materials {

Vec3 reflect(Vec3 dir) {
	//A3T5 Materials - reflect helper
	// Reflect dir about the normal (0,1,0)
	return Vec3(-dir.x, dir.y, -dir.z);
}

Vec3 refract(Vec3 out_dir, float index_of_refraction, bool& was_internal) {
	//A3T5 Materials - refract helper

	// Use Snell's Law to refract out_dir through the surface.
	// Return the refracted direction. Set was_internal to true if
	// refraction does not occur due to total internal reflection,
	// and false otherwise.

	// The surface normal is (0,1,0)
	Vec3 normal = Vec3(0.f, 1.f, 0.f);
	float cosTheta_i = dot(normal, out_dir);
	float eta_i = 1.f;
	float eta_t = index_of_refraction;
	// leaving
	if(cosTheta_i < 0) {
		std::swap(eta_i, eta_t);
		normal = -normal;
		cosTheta_i = -cosTheta_i;
	}
	float eta = eta_i / eta_t;
	float sin2Theta_i = 1.f - cosTheta_i * cosTheta_i;
	float sin2Theta_t = eta * eta * sin2Theta_i;
	if(sin2Theta_t > 1.f) {
		was_internal = true;
		return Vec3{};
	}
	float cosTheta_t = std::sqrt(1.f - sin2Theta_t);
	was_internal = false;
	return eta * (-out_dir) + (eta * cosTheta_i - cosTheta_t) * normal;
}

float schlick(Vec3 in_dir, float index_of_refraction) {
	//A3T5 Materials - Schlick's approximation helper
	// Assume normal is (0,1,0)
	float cos_theta = fabsf(in_dir.y);
	float r0 = (1.0f - index_of_refraction) / (1.0f + index_of_refraction);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);
}

Spectrum Lambertian::evaluate(Vec3 out, Vec3 in, Vec2 uv) const {
	//A3T4: Materials - Lambertian BSDF evaluation

	// Compute the ratio of outgoing/incoming radiance when light from in_dir
	// is reflected through out_dir: (albedo / PI_F) * cos(theta).
	// Note that for Scotty3D, y is the 'up' direction.

	float cos_theta = in.y;
	if (cos_theta <= 0.0f) return Spectrum{};

	Spectrum albedo_val = albedo.lock()->evaluate(uv);
	return albedo_val * cos_theta / PI_F;
}

Scatter Lambertian::scatter(RNG &rng, Vec3 out, Vec2 uv) const {
	//A3T4: Materials - Lambertian BSDF scattering
	//Select a scattered light direction at random from the Lambertian BSDF

	Samplers::Hemisphere::Cosine sampler;

	Scatter ret;
	// Sample the direction the light was scattered from, from a cosine-weighted hemisphere distribution:
	ret.direction = sampler.sample(rng);

	// Compute the attenuation of the light using Lambertian::evaluate:
	ret.attenuation = evaluate(out, ret.direction, uv);

	return ret;
}

float Lambertian::pdf(Vec3 out, Vec3 in) const {
	//A3T4: Materials - Lambertian BSDF probability density function
	Samplers::Hemisphere::Cosine sampler;
	return sampler.pdf(in);
}

Spectrum Lambertian::emission(Vec2 uv) const {
	return {};
}

std::weak_ptr<Texture> Lambertian::display() const {
	return albedo;
}

void Lambertian::for_each(const std::function<void(std::weak_ptr<Texture>&)>& f) {
	f(albedo);
}

Spectrum Mirror::evaluate(Vec3 out, Vec3 in, Vec2 uv) const {
	return {};
}

Scatter Mirror::scatter(RNG &rng, Vec3 out, Vec2 uv) const {
	//A3T5: mirror

	// Use reflect to compute the new direction
	// Don't forget that this is a discrete material!
	// Similar to albedo, reflectance represents the ratio of incoming light to reflected light

    Scatter ret;
    ret.direction = reflect(out);
    ret.attenuation = reflectance.lock()->evaluate(uv);
    return ret;
}

float Mirror::pdf(Vec3 out, Vec3 in) const {
	return 0.0f;
}

Spectrum Mirror::emission(Vec2 uv) const {
	return {};
}

std::weak_ptr<Texture> Mirror::display() const {
	return reflectance;
}

void Mirror::for_each(const std::function<void(std::weak_ptr<Texture>&)>& f) {
	f(reflectance);
}

Spectrum Refract::evaluate(Vec3 out, Vec3 in, Vec2 uv) const {
	return {};
}

Scatter Refract::scatter(RNG &rng, Vec3 out, Vec2 uv) const {
	//A3T5 - refract

	// Use refract to determine the new direction - what happens in the total internal reflection case?
    // Be wary of your eta1/eta2 ratio - are you entering or leaving the surface?
	// Don't forget that this is a discrete material!
	// For attenuation, be sure to take a look at the Specular Transimission section of the PBRT textbook for a derivation
	//  You do not need to scale by the Fresnel Coefficient - you'll only need to account for the correct ratio of indices of refraction

    Scatter ret;
    bool was_internal = false;
    ret.direction = refract(out, ior, was_internal);
    if (was_internal) {
        // Total internal reflection, no transmission
        ret.attenuation = Spectrum{};
    } else {
        // Scale by eta^2 (PBRT specular transmission)
        float eta = (out.y > 0.0f) ? 1.0f / ior : ior;
        ret.attenuation = transmittance.lock()->evaluate(uv) * (eta * eta);
    }
    return ret;
}

float Refract::pdf(Vec3 out, Vec3 in) const {
	return 0.0f;
}

Spectrum Refract::emission(Vec2 uv) const {
	return {};
}

bool Refract::is_emissive() const {
	return false;
}

bool Refract::is_specular() const {
	return true;
}

bool Refract::is_sided() const {
	return true;
}

std::weak_ptr<Texture> Refract::display() const {
	return transmittance;
}

void Refract::for_each(const std::function<void(std::weak_ptr<Texture>&)>& f) {
	f(transmittance);
}

Spectrum Glass::evaluate(Vec3 out, Vec3 in, Vec2 uv) const {
	return {};
}

Scatter Glass::scatter(RNG &rng, Vec3 out, Vec2 uv) const {
	//A3T5 - glass

    // (1) Compute Fresnel coefficient. Tip: Schlick's approximation.
    // (2) Reflect or refract probabilistically based on Fresnel coefficient. Tip: RNG::coin_flip
    // (3) Compute attenuation based on reflectance or transmittance

    // Be wary of your eta1/eta2 ratio - are you entering or leaving the surface?
    // What happens upon total internal reflection?
    // When debugging Glass, it may be useful to compare to a pure-refraction BSDF
	// For attenuation, be sure to take a look at the Specular Transimission section of the PBRT textbook for a derivation
	//  You do not need to scale by the Fresnel Coefficient - you'll only need to account for the correct ratio of indices of refraction

    Scatter ret;
    bool was_internal = false;
    float F = schlick(out, ior);
    if (rng.coin_flip(F)) {
        // Reflect
        ret.direction = reflect(out);
        ret.attenuation = reflectance.lock()->evaluate(uv);
    } else {
        // Refract
        ret.direction = refract(out, ior, was_internal);
        if (was_internal) {
            // Total internal reflection, fallback to reflection
            ret.direction = reflect(out);
            ret.attenuation = reflectance.lock()->evaluate(uv);
        } else {
            float eta = (out.y > 0.0f) ? 1.0f / ior : ior;
            ret.attenuation = transmittance.lock()->evaluate(uv) * (eta * eta);
        }
    }
    return ret;
}

float Glass::pdf(Vec3 out, Vec3 in) const {
	return 0.0f;
}

Spectrum Glass::emission(Vec2 uv) const {
	return {};
}

bool Glass::is_emissive() const {
	return false;
}

bool Glass::is_specular() const {
	return true;
}

bool Glass::is_sided() const {
	return true;
}

std::weak_ptr<Texture> Glass::display() const {
	return transmittance;
}

void Glass::for_each(const std::function<void(std::weak_ptr<Texture>&)>& f) {
	f(reflectance);
	f(transmittance);
}

Spectrum Emissive::evaluate(Vec3 out, Vec3 in, Vec2 uv) const {
	return {};
}

Scatter Emissive::scatter(RNG &rng, Vec3 out, Vec2 uv) const {
	Scatter ret;
	ret.direction = {};
	ret.attenuation = {};
	return ret;
}

float Emissive::pdf(Vec3 out, Vec3 in) const {
	return 0.0f;
}

Spectrum Emissive::emission(Vec2 uv) const {
	return emissive.lock()->evaluate(uv);
}

bool Emissive::is_emissive() const {
	return true;
}

bool Emissive::is_specular() const {
	return true;
}

bool Emissive::is_sided() const {
	return false;
}

std::weak_ptr<Texture> Emissive::display() const {
	return emissive;
}

void Emissive::for_each(const std::function<void(std::weak_ptr<Texture>&)>& f) {
	f(emissive);
}

} // namespace Materials

bool operator!=(const Materials::Lambertian& a, const Materials::Lambertian& b) {
	return a.albedo.lock() != b.albedo.lock();
}

bool operator!=(const Materials::Mirror& a, const Materials::Mirror& b) {
	return a.reflectance.lock() != b.reflectance.lock();
}

bool operator!=(const Materials::Refract& a, const Materials::Refract& b) {
	return a.transmittance.lock() != b.transmittance.lock() || a.ior != b.ior;
}

bool operator!=(const Materials::Glass& a, const Materials::Glass& b) {
	return a.reflectance.lock() != b.reflectance.lock() ||
	       a.transmittance.lock() != b.transmittance.lock() || a.ior != b.ior;
}

bool operator!=(const Materials::Emissive& a, const Materials::Emissive& b) {
	return a.emissive.lock() != b.emissive.lock();
}

bool operator!=(const Material& a, const Material& b) {
	if (a.material.index() != b.material.index()) return false;
	return std::visit(
		[&](const auto& material) {
			return material != std::get<std::decay_t<decltype(material)>>(b.material);
		},
		a.material);
}
