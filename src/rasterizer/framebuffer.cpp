#include "framebuffer.h"
#include "../util/hdr_image.h"
#include "sample_pattern.h"

Framebuffer::Framebuffer(uint32_t width_, uint32_t height_, SamplePattern const& sample_pattern_)
	: width(width_), height(height_), sample_pattern(sample_pattern_) {

	// check that framebuffer isn't larger than allowed:
	if (width > MaxWidth || height > MaxHeight) {
		throw std::runtime_error("Framebuffer size (" + std::to_string(width) + "x" +
		                         std::to_string(height) + ") exceeds maximum allowed (" +
		                         std::to_string(MaxWidth) + "x" + std::to_string(MaxHeight) + ").");
	}
	// check that framebuffer size is even:
	if (width % 2 != 0 || height % 2 != 0) {
		throw std::runtime_error("Framebuffer size (" + std::to_string(width) + "x" +
		                         std::to_string(height) + ") is not even.");
	}

	uint32_t samples =
		width * height * static_cast<uint32_t>(sample_pattern.centers_and_weights.size());

	// allocate storage for color and depth samples:
	colors.assign(samples, Spectrum{0.0f, 0.0f, 0.0f});
	depths.assign(samples, 1.0f);
}

HDR_Image Framebuffer::resolve_colors() const {
	// Support sample patterns with more than one sample.

	HDR_Image image(width, height);
	size_t num_samples = sample_pattern.centers_and_weights.size();

	for (uint32_t y = 0; y < height; ++y) {
		for (uint32_t x = 0; x < width; ++x) {
			Spectrum sum{0.0f, 0.0f, 0.0f};
			float total_weight = 0.0f;
			for (size_t s = 0; s < num_samples; ++s) {
				float weight = sample_pattern.centers_and_weights[s].z;
				sum += color_at(x, y, s) * weight;
				total_weight += weight;
			}
			if (total_weight > 0.0f) {
				sum = sum / total_weight;
			}
			image.at(x, y) = sum;
		}
	}

	return image;
}
