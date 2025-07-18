
#include "texture.h"

#include <iostream>

namespace Textures {


Spectrum sample_nearest(HDR_Image const &image, Vec2 uv) {
	//clamp texture coordinates, convert to [0,w]x[0,h] pixel space:
	float x = image.w * std::clamp(uv.x, 0.0f, 1.0f);
	float y = image.h * std::clamp(uv.y, 0.0f, 1.0f);

	//the pixel with the nearest center is the pixel that contains (x,y):
	int32_t ix = int32_t(std::floor(x));
	int32_t iy = int32_t(std::floor(y));

	//texture coordinates of (1,1) map to (w,h), and need to be reduced:
	ix = std::min(ix, int32_t(image.w) - 1);
	iy = std::min(iy, int32_t(image.h) - 1);

	return image.at(ix, iy);
}

Spectrum sample_bilinear(HDR_Image const &image, Vec2 uv) {
	// A1T6: sample_bilinear
	
	// Bilinear sampling
	float x = uv.x * image.w - 0.5f;
	float y = uv.y * image.h - 0.5f;

	int x0 = int(std::floor(x));
	int y0 = int(std::floor(y));
	int x1 = x0 + 1;
	int y1 = y0 + 1;

	float sx = x - x0;
	float sy = y - y0;

	x0 = std::clamp(x0, 0, int(image.w) - 1);
	y0 = std::clamp(y0, 0, int(image.h) - 1);
	x1 = std::clamp(x1, 0, int(image.w) - 1);
	y1 = std::clamp(y1, 0, int(image.h) - 1);

	Spectrum c00 = image.at(x0, y0);
	Spectrum c10 = image.at(x1, y0);
	Spectrum c01 = image.at(x0, y1);
	Spectrum c11 = image.at(x1, y1);

	Spectrum c0 = c00 * (1.0f - sx) + c10 * sx;
	Spectrum c1 = c01 * (1.0f - sx) + c11 * sx;
	return c0 * (1.0f - sy) + c1 * sy;
}

Spectrum sample_trilinear(HDR_Image const &base, std::vector< HDR_Image > const &levels, Vec2 uv, float lod) {
	// Task 7: Implement trilinear filtering

	// Return magenta for invalid level
	float level = std::max(std::min(lod, float(levels.size())), 0.0f);
	int ld = int(std::floor(level));
	int hd = ld + 1;

	// If ld is out of bounds, return magenta
	if (ld >= int(levels.size()) + 1) {
		return Spectrum(1.0f, 0.0f, 1.0f);
	}

	// Clamp hd to valid range
	if (hd > int(levels.size())) hd = int(levels.size());

	// Select images for ld and hd
	const HDR_Image& img_ld = (ld == 0) ? base : levels[ld - 1];
	const HDR_Image& img_hd = (hd == 0) ? base : levels[hd - 1];

	Spectrum lc = sample_bilinear(img_ld, uv);
	Spectrum hc = sample_bilinear(img_hd, uv);

	// No alpha channel in Spectrum, so skip premultiplied alpha logic

	float dd = level - float(ld);
	Spectrum mix = lc * (1.0f - dd) + hc * dd;
	return mix;
}

/*
 * generate_mipmap- generate mipmap levels from a base image.
 *  base: the base image
 *  levels: pointer to vector of levels to fill (must not be null)
 *
 * generates a stack of levels [1,n] of sizes w_i, h_i, where:
 *   w_i = max(1, floor(w_{i-1})/2)
 *   h_i = max(1, floor(h_{i-1})/2)
 *  with:
 *   w_0 = base.w
 *   h_0 = base.h
 *  and n is the smalles n such that w_n = h_n = 1
 *
 * each level should be calculated by downsampling a blurred version
 * of the previous level to remove high-frequency detail.
 *
 */
void generate_mipmap(HDR_Image const &base, std::vector< HDR_Image > *levels_) {
	assert(levels_);
	auto &levels = *levels_;


	{ // allocate sublevels sufficient to scale base image all the way to 1x1:
		int32_t num_levels = static_cast<int32_t>(std::log2(std::max(base.w, base.h)));
		assert(num_levels >= 0);

		levels.clear();
		levels.reserve(num_levels);

		uint32_t width = base.w;
		uint32_t height = base.h;
		for (int32_t i = 0; i < num_levels; ++i) {
			assert(!(width == 1 && height == 1)); //would have stopped before this if num_levels was computed correctly

			width = std::max(1u, width / 2u);
			height = std::max(1u, height / 2u);

			levels.emplace_back(width, height);
		}
		assert(width == 1 && height == 1);
		assert(levels.size() == uint32_t(num_levels));
	}

	//now fill in the levels using a helper:
	//downsample:
	// fill in dst to represent the low-frequency component of src
	auto downsample = [](HDR_Image const &src, HDR_Image &dst) {
		//dst is half the size of src in each dimension:
		assert(std::max(1u, src.w / 2u) == dst.w);
		assert(std::max(1u, src.h / 2u) == dst.h);
		// A1T6: generate

		//Be aware that the alignment of the samples in dst and src will be different depending on whether the image is even or odd.
		// For each pixel in dst, average the corresponding 2x2 block in src
		for (uint32_t y = 0; y < dst.h; ++y) {
			for (uint32_t x = 0; x < dst.w; ++x) {
				// Compute the coordinates in src
				uint32_t x0 = x * 2;
				uint32_t y0 = y * 2;

				// Collect up to 4 samples (handle edge cases)
				Spectrum sum(0.0f);
				int count = 0;
				for (uint32_t dy = 0; dy < 2; ++dy) {
					for (uint32_t dx = 0; dx < 2; ++dx) {
						uint32_t sx = x0 + dx;
						uint32_t sy = y0 + dy;
						if (sx < src.w && sy < src.h) {
							sum += src.at(sx, sy);
							++count;
						}
					}
				}
				dst.at(x, y) = sum * (1.0f / float(count));
			}
		}
	};

	std::cout << "Regenerating mipmap (" << levels.size() << " levels): [" << base.w << "x" << base.h << "]";
	std::cout.flush();
	for (uint32_t i = 0; i < levels.size(); ++i) {
		HDR_Image const &src = (i == 0 ? base : levels[i-1]);
		HDR_Image &dst = levels[i];
		std::cout << " -> [" << dst.w << "x" << dst.h << "]"; std::cout.flush();

		downsample(src, dst);
	}
	std::cout << std::endl;
	
}

Image::Image(Sampler sampler_, HDR_Image const &image_) {
	sampler = sampler_;
	image = image_.copy();
	update_mipmap();
}

Spectrum Image::evaluate(Vec2 uv, float lod) const {
	if (image.w == 0 && image.h == 0) return Spectrum();
	if (sampler == Sampler::nearest) {
		return sample_nearest(image, uv);
	} else if (sampler == Sampler::bilinear) {
		return sample_bilinear(image, uv);
	} else {
		return sample_trilinear(image, levels, uv, lod);
	}
}

void Image::update_mipmap() {
	if (sampler == Sampler::trilinear) {
		generate_mipmap(image, &levels);
	} else {
		levels.clear();
	}
}

GL::Tex2D Image::to_gl() const {
	return image.to_gl(1.0f);
}

void Image::make_valid() {
	update_mipmap();
}

Spectrum Constant::evaluate(Vec2 uv, float lod) const {
	return color * scale;
}

} // namespace Textures
bool operator!=(const Textures::Constant& a, const Textures::Constant& b) {
	return a.color != b.color || a.scale != b.scale;
}

bool operator!=(const Textures::Image& a, const Textures::Image& b) {
	return a.image != b.image;
}

bool operator!=(const Texture& a, const Texture& b) {
	if (a.texture.index() != b.texture.index()) return false;
	return std::visit(
		[&](const auto& data) { return data != std::get<std::decay_t<decltype(data)>>(b.texture); },
		a.texture);
}
