// clang-format off
#include "pipeline.h"

#include <iostream>

#include "../lib/log.h"
#include "../lib/mathlib.h"
#include "framebuffer.h"
#include "sample_pattern.h"
template<PrimitiveType primitive_type, class Program, uint32_t flags>
void Pipeline<primitive_type, Program, flags>::run(std::vector<Vertex> const& vertices,
												   typename Program::Parameters const& parameters,
												   Framebuffer* framebuffer_) {
	// Framebuffer must be non-null:
	assert(framebuffer_);
	auto& framebuffer = *framebuffer_;

	// A1T7: sample loop
	std::vector<Vec3> const& samples = framebuffer.sample_pattern.centers_and_weights;
	uint32_t sample_count = static_cast<uint32_t>(samples.size());

	// For each sample location:
	for (uint32_t s = 0; s < sample_count; ++s) {
		std::vector<ShadedVertex> shaded_vertices;
		shaded_vertices.reserve(vertices.size());

		//--------------------------
		// shade vertices:
		for (auto const& v : vertices) {
			ShadedVertex sv;
			Program::shade_vertex(parameters, v.attributes, &sv.clip_position, &sv.attributes);
			shaded_vertices.emplace_back(sv);
		}

		//--------------------------
		// assemble + clip + homogeneous divide vertices:
		std::vector<ClippedVertex> clipped_vertices;

		// reserve some space to avoid reallocations later:
		if constexpr (primitive_type == PrimitiveType::Lines) {
			clipped_vertices.reserve(shaded_vertices.size());
		} else if constexpr (primitive_type == PrimitiveType::Triangles) {
			clipped_vertices.reserve(shaded_vertices.size() * 8);
		}

		//coefficients to map from clip coordinates to framebuffer (i.e., "viewport") coordinates:
		Vec3 const clip_to_fb_scale = Vec3{
			framebuffer.width / 2.0f,
			framebuffer.height / 2.0f,
			0.5f
		};
		Vec3 const clip_to_fb_offset = Vec3{
			0.5f * framebuffer.width,
			0.5f * framebuffer.height,
			0.5f
		};

		// helper used to put output of clipping functions into clipped_vertices:
		auto emit_vertex = [&](ShadedVertex const& sv) {
			ClippedVertex cv;
			float inv_w = 1.0f / sv.clip_position.w;
			cv.fb_position = clip_to_fb_scale * inv_w * sv.clip_position.xyz() + clip_to_fb_offset;
			// Shift by sample offset (A1T7)
			cv.fb_position.x += samples[s].x - 0.5f;
			cv.fb_position.y += samples[s].y - 0.5f;
			cv.inv_w = inv_w;
			cv.attributes = sv.attributes;
			clipped_vertices.emplace_back(cv);
		};

		// actually do clipping:
		if constexpr (primitive_type == PrimitiveType::Lines) {
			for (uint32_t i = 0; i + 1 < shaded_vertices.size(); i += 2) {
				clip_line(shaded_vertices[i], shaded_vertices[i + 1], emit_vertex);
			}
		} else if constexpr (primitive_type == PrimitiveType::Triangles) {
			for (uint32_t i = 0; i + 2 < shaded_vertices.size(); i += 3) {
				clip_triangle(shaded_vertices[i], shaded_vertices[i + 1], shaded_vertices[i + 2], emit_vertex);
			}
		} else {
			static_assert(primitive_type == PrimitiveType::Lines, "Unsupported primitive type.");
		}

		//--------------------------
		// rasterize primitives:

		std::vector<Fragment> fragments;

		// helper used to put output of rasterization functions into fragments:
		auto emit_fragment = [&](Fragment const& f) { fragments.emplace_back(f); };

		// actually do rasterization:
		if constexpr (primitive_type == PrimitiveType::Lines) {
			for (uint32_t i = 0; i + 1 < clipped_vertices.size(); i += 2) {
				rasterize_line(clipped_vertices[i], clipped_vertices[i + 1], emit_fragment);
			}
		} else if constexpr (primitive_type == PrimitiveType::Triangles) {
			for (uint32_t i = 0; i + 2 < clipped_vertices.size(); i += 3) {
				rasterize_triangle(clipped_vertices[i], clipped_vertices[i + 1], clipped_vertices[i + 2], emit_fragment);
			}
		} else {
			static_assert(primitive_type == PrimitiveType::Lines, "Unsupported primitive type.");
		}

		//--------------------------
		// depth test + shade + blend fragments:
		uint32_t out_of_range = 0;
		for (auto const& f : fragments) {
			int32_t x = (int32_t)std::floor(f.fb_position.x);
			int32_t y = (int32_t)std::floor(f.fb_position.y);

			if (x < 0 || (uint32_t)x >= framebuffer.width ||
				y < 0 || (uint32_t)y >= framebuffer.height) {
				++out_of_range;
				continue;
			}

			// Use the correct sample index for multisample (A1T7)
			float& fb_depth = framebuffer.depth_at(x, y, s);
			Spectrum& fb_color = framebuffer.color_at(x, y, s);

			if constexpr ((flags & PipelineMask_Depth) == Pipeline_Depth_Always) {
				// always pass
			} else if constexpr ((flags & PipelineMask_Depth) == Pipeline_Depth_Never) {
				continue;
			} else if constexpr ((flags & PipelineMask_Depth) == Pipeline_Depth_Less) {
				if (f.fb_position.z >= fb_depth) {
					continue;
				}
			} else {
				static_assert((flags & PipelineMask_Depth) <= Pipeline_Depth_Always, "Unknown depth test flag.");
			}

			if constexpr (!(flags & Pipeline_DepthWriteDisableBit)) {
				fb_depth = f.fb_position.z;
			}

			ShadedFragment sf;
			sf.fb_position = f.fb_position;
			Program::shade_fragment(parameters, f.attributes, f.derivatives, &sf.color, &sf.opacity);

			if constexpr (!(flags & Pipeline_ColorWriteDisableBit)) {
				if constexpr ((flags & PipelineMask_Blend) == Pipeline_Blend_Replace) {
					fb_color = sf.color;
				} else if constexpr ((flags & PipelineMask_Blend) == Pipeline_Blend_Add) {
					fb_color += sf.color * sf.opacity;
				} else if constexpr ((flags & PipelineMask_Blend) == Pipeline_Blend_Over) {
					fb_color = sf.color * sf.opacity + fb_color * (1.0f - sf.opacity);
				} else {
					static_assert((flags & PipelineMask_Blend) <= Pipeline_Blend_Over, "Unknown blending flag.");
				}
			}
		}
		if (out_of_range > 0) {
			if constexpr (primitive_type == PrimitiveType::Lines) {
				warn("Produced %d fragments outside framebuffer; this indicates something is likely "
					"wrong with the clip_line function.",
					out_of_range);
			} else if constexpr (primitive_type == PrimitiveType::Triangles) {
				warn("Produced %d fragments outside framebuffer; this indicates something is likely "
					"wrong with the clip_triangle function.",
					out_of_range);
			}
		}
	}
}

// -------------------------------------------------------------------------
// clipping functions

// helper to interpolate between vertices:
template<PrimitiveType p, class P, uint32_t F>
auto Pipeline<p, P, F>::lerp(ShadedVertex const& a, ShadedVertex const& b, float t) -> ShadedVertex {
	ShadedVertex ret;
	ret.clip_position = (b.clip_position - a.clip_position) * t + a.clip_position;
	for (uint32_t i = 0; i < ret.attributes.size(); ++i) {
		ret.attributes[i] = (b.attributes[i] - a.attributes[i]) * t + a.attributes[i];
	}
	return ret;
}

/*
 * clip_line - clip line to portion with -w <= x,y,z <= w, emit vertices of clipped line (if non-empty)
 *  	va, vb: endpoints of line
 *  	emit_vertex: call to produce truncated line
 *
 * If clipping shortens the line, attributes of the shortened line should respect the pipeline's interpolation mode.
 * 
 * If no portion of the line remains after clipping, emit_vertex will not be called.
 *
 * The clipped line should have the same direction as the full line.
 */
template<PrimitiveType p, class P, uint32_t flags>
void Pipeline<p, P, flags>::clip_line(ShadedVertex const& va, ShadedVertex const& vb,
                                      std::function<void(ShadedVertex const&)> const& emit_vertex) {
	// Determine portion of line over which:
	// 		pt = (b-a) * t + a
	//  	-pt.w <= pt.x <= pt.w
	//  	-pt.w <= pt.y <= pt.w
	//  	-pt.w <= pt.z <= pt.w
	// ... as a range [min_t, max_t]:

	float min_t = 0.0f;
	float max_t = 1.0f;

	// want to set range of t for a bunch of equations like:
	//    a.x + t * ba.x <= a.w + t * ba.w
	// so here's a helper:
	auto clip_range = [&min_t, &max_t](float l, float dl, float r, float dr) {
		// restrict range such that:
		// l + t * dl <= r + t * dr
		// re-arranging:
		//  l - r <= t * (dr - dl)
		if (dr == dl) {
			// want: l - r <= 0
			if (l - r > 0.0f) {
				// works for none of range, so make range empty:
				min_t = 1.0f;
				max_t = 0.0f;
			}
		} else if (dr > dl) {
			// since dr - dl is positive:
			// want: (l - r) / (dr - dl) <= t
			min_t = std::max(min_t, (l - r) / (dr - dl));
		} else { // dr < dl
			// since dr - dl is negative:
			// want: (l - r) / (dr - dl) >= t
			max_t = std::min(max_t, (l - r) / (dr - dl));
		}
	};

	// local names for clip positions and their difference:
	Vec4 const& a = va.clip_position;
	Vec4 const& b = vb.clip_position;
	Vec4 const ba = b - a;

	// -a.w - t * ba.w <= a.x + t * ba.x <= a.w + t * ba.w
	clip_range(-a.w, -ba.w, a.x, ba.x);
	clip_range(a.x, ba.x, a.w, ba.w);
	// -a.w - t * ba.w <= a.y + t * ba.y <= a.w + t * ba.w
	clip_range(-a.w, -ba.w, a.y, ba.y);
	clip_range(a.y, ba.y, a.w, ba.w);
	// -a.w - t * ba.w <= a.z + t * ba.z <= a.w + t * ba.w
	clip_range(-a.w, -ba.w, a.z, ba.z);
	clip_range(a.z, ba.z, a.w, ba.w);

	if (min_t < max_t) {
		if (min_t == 0.0f) {
			emit_vertex(va);
		} else {
			ShadedVertex out = lerp(va, vb, min_t);
			// don't interpolate attributes if in flat shading mode:
			if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Flat) {
				out.attributes = va.attributes;
			}
			emit_vertex(out);
		}
		if (max_t == 1.0f) {
			emit_vertex(vb);
		} else {
			ShadedVertex out = lerp(va, vb, max_t);
			// don't interpolate attributes if in flat shading mode:
			if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Flat) {
				out.attributes = va.attributes;
			}
			emit_vertex(out);
		}
	}
}

/*
 * clip_triangle - clip triangle to portion with -w <= x,y,z <= w, emit resulting shape as triangles (if non-empty)
 *  	va, vb, vc: vertices of triangle
 *  	emit_vertex: call to produce clipped triangles (three calls per triangle)
 *
 * If clipping truncates the triangle, attributes of the new vertices should respect the pipeline's interpolation mode.
 * 
 * If no portion of the triangle remains after clipping, emit_vertex will not be called.
 *
 * The clipped triangle(s) should have the same winding order as the full triangle.
 */
template<PrimitiveType p, class P, uint32_t flags>
void Pipeline<p, P, flags>::clip_triangle(
	ShadedVertex const& va, ShadedVertex const& vb, ShadedVertex const& vc,
	std::function<void(ShadedVertex const&)> const& emit_vertex) {
	// A1EC: clip_triangle
	// TODO: correct code!
	emit_vertex(va);
	emit_vertex(vb);
	emit_vertex(vc);
}

// -------------------------------------------------------------------------
// rasterization functions

/*
 * rasterize_line:
 * calls emit_fragment( frag ) for every pixel "covered" by the line (va.fb_position.xy, vb.fb_position.xy).
 *
 *    a pixel (x,y) is "covered" by the line if it exits the inscribed diamond:
 * 
 *        (x+0.5,y+1)
 *        /        \
 *    (x,y+0.5)  (x+1,y+0.5)
 *        \        /
 *         (x+0.5,y)
 *
 *    to avoid ambiguity, we consider diamonds to contain their left and bottom points
 *    but not their top and right points. 
 * 
 * 	  since 45 degree lines breaks this rule, our rule in general is to rasterize the line as if its
 *    endpoints va and vb were at va + (e, e^2) and vb + (e, e^2) where no smaller nonzero e produces 
 *    a different rasterization result. 
 *    We will not explicitly check for 45 degree lines along the diamond edges (this will be extra credit),
 *    but you should be able to handle 45 degree lines in every other case (such as starting from pixel centers)
 *
 * for each such diamond, pass Fragment frag to emit_fragment, with:
 *  - frag.fb_position.xy set to the center (x+0.5,y+0.5)
 *  - frag.fb_position.z interpolated linearly between va.fb_position.z and vb.fb_position.z
 *  - frag.attributes set to va.attributes (line will only be used in Interp_Flat mode)
 *  - frag.derivatives set to all (0,0)
 *
 * when interpolating the depth (z) for the fragments, you may use any depth the line takes within the pixel
 * (i.e., you don't need to interpolate to, say, the closest point to the pixel center)
 *
 * If you wish to work in fixed point, check framebuffer.h for useful information about the framebuffer's dimensions.
 */
template<PrimitiveType p, class P, uint32_t flags>
void Pipeline<p, P, flags>::rasterize_line(
	ClippedVertex const& va, ClippedVertex const& vb,
	std::function<void(Fragment const&)> const& emit_fragment) {
	if constexpr ((flags & PipelineMask_Interp) != Pipeline_Interp_Flat) {
		assert(0 && "rasterize_line should only be invoked in flat interpolation mode.");
	}
	// A1T2: rasterize_line

	// this function!
	// The OpenGL specification section 3.5 may also come in handy.
	float delta_x = vb.fb_position.x - va.fb_position.x;
	float delta_y = vb.fb_position.y - va.fb_position.y;
	int i, j;
	if (delta_x > delta_y) {
		i = 0;
		j = 1;
	} else {
		i = 1;
		j = 0;
	}
	Vec3 pa = va.fb_position;
	Vec3 pb = vb.fb_position;
	if (pa[i] > pb[i]) {
		std::swap(pa, pb);
	}
	int t1 = std::floor(pa[i]);
	int t2 = std::floor(pb[i]); 
	if (t1>=t2)
	{
		return; // nothing to rasterize
	}
	
	Fragment frag;
	for (float u = t1; u <= t2; u++) {
		float w = ((u + 0.5f) - pa[i]) / (pb[i] - pa[i]);
		float v = w * (pb[j] - pa[j]) + pa[j];
		frag.fb_position.data[i] = std::floor(u) + 0.5f;
		frag.fb_position.data[j] = std::floor(v) + 0.5f; 
		frag.fb_position.data[2] = w * (pb[2] - pa[2]) + pa[2];
		frag.attributes = va.attributes;
		frag.derivatives.fill(Vec2(0.0f, 0.0f));
		if (abs(pb[0] - frag.fb_position.data[0]) + abs(pb[1] - frag.fb_position.data[1]) >= 0.5f 
		    || (pb[0] == frag.fb_position.data[0] + 0.5f && pb[1] == frag.fb_position.data[1])
			|| (pb[0] == frag.fb_position.data[0] && pb[1] == frag.fb_position.data[2] + 0.5f)
		) {
			emit_fragment(frag);
		}
	}
}

/*
 * rasterize_triangle(a,b,c,emit) calls 'emit(frag)' at every location
 *  	(x+0.5,y+0.5) (where x,y are integers) covered by triangle (a,b,c).
 *
 * The emitted fragment should have:
 * - frag.fb_position.xy = (x+0.5, y+0.5)
 * - frag.fb_position.z = linearly interpolated fb_position.z from a,b,c (NOTE: does not depend on Interp mode!)
 * - frag.attributes = depends on Interp_* flag in flags:
 *   - if Interp_Flat: copy from va.attributes
 *   - if Interp_Smooth: interpolate as if (a,b,c) is a 2D triangle flat on the screen
 *   - if Interp_Correct: use perspective-correct interpolation
 * - frag.derivatives = derivatives w.r.t. fb_position.x and fb_position.y of the first frag.derivatives.size() attributes.
 *
 * Notes on derivatives:
 * 	The derivatives are partial derivatives w.r.t. screen locations. That is:
 *    derivatives[i].x = d/d(fb_position.x) attributes[i]
 *    derivatives[i].y = d/d(fb_position.y) attributes[i]
 *  You may compute these derivatives analytically or numerically.
 *
 *  See section 8.12.1 "Derivative Functions" of the GLSL 4.20 specification for some inspiration. (*HOWEVER*, the spec is solving a harder problem, and also nothing in the spec is binding on your implementation)
 *
 *  One approach is to rasterize blocks of four fragments and use forward and backward differences to compute derivatives.
 *  To assist you in this approach, keep in mind that the framebuffer size is *guaranteed* to be even. (see framebuffer.h)
 *
 * Notes on coverage:
 *  If two triangles are on opposite sides of the same edge, and a
 *  fragment center lies on that edge, rasterize_triangle should
 *  make sure that exactly one of the triangles emits that fragment.
 *  (Otherwise, speckles or cracks can appear in the final render.)
 * 
 *  For degenerate (co-linear) triangles, you may consider them to not be on any side of an edge.
 * 	Thus, even if two degnerate triangles share an edge that contains a fragment center, you don't need to emit it.
 *  You will not lose points for doing something reasonable when handling this case
 *
 *  This is pretty tricky to get exactly right!
 *
 */
template<PrimitiveType p, class P, uint32_t flags>
void Pipeline<p, P, flags>::rasterize_triangle(
	ClippedVertex const& va, ClippedVertex const& vb, ClippedVertex const& vc,
	std::function<void(Fragment const&)> const& emit_fragment) {
	// NOTE: it is okay to restructure this function to allow these tasks to use the
	//  same code paths. Be aware, however, that all of them need to remain working!
	//  (e.g., if you break Flat while implementing Correct, you won't get points
	//   for Flat.)
	Vec3 const& a = va.fb_position;
	Vec3 const& b = vb.fb_position;
	Vec3 const& c = vc.fb_position;

	float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
	if (area == 0.0f) return;

	Vec2 edge0 = Vec2(b.x - a.x, b.y - a.y);
	Vec2 edge1 = Vec2(c.x - b.x, c.y - b.y);
	Vec2 edge2 = Vec2(a.x - c.x, a.y - c.y);
	auto is_start_edge = [](Vec2 const& e) -> bool {
		return (e.y > 0.0f) || (e.y == 0.0f && e.x > 0.0f);
	};
	bool is_start_edge0 = is_start_edge(edge0);
	bool is_start_edge1 = is_start_edge(edge1);
	bool is_start_edge2 = is_start_edge(edge2);

	float min_x = std::min({a.x, b.x, c.x});
	float max_x = std::max({a.x, b.x, c.x});
	float min_y = std::min({a.y, b.y, c.y});
	float max_y = std::max({a.y, b.y, c.y});

	int32_t min_ix = static_cast<int32_t>(std::floor(min_x));
	int32_t max_ix = static_cast<int32_t>(std::ceil(max_x));
	int32_t min_iy = static_cast<int32_t>(std::floor(min_y));
	int32_t max_iy = static_cast<int32_t>(std::ceil(max_y));

	if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Flat) {
		// A1T3: flat triangles
		for (int32_t iy = min_iy; iy <= max_iy; ++iy) {
			for (int32_t ix = min_ix; ix <= max_ix; ++ix) {
				Vec2 q = Vec2(ix + 0.5f, iy + 0.5f);

				float w0 = (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x);
				float w1 = (c.x - b.x) * (q.y - b.y) - (c.y - b.y) * (q.x - b.x);
				float w2 = (a.x - c.x) * (q.y - c.y) - (a.y - c.y) * (q.x - c.x);

				bool inside;
				if (area > 0.0f) {
					inside = (w0 >= 0) && (w1 >= 0) && (w2 >= 0);
					if (w0 == 0.0f) inside = inside && is_start_edge0;
					if (w1 == 0.0f) inside = inside && is_start_edge1;
					if (w2 == 0.0f) inside = inside && is_start_edge2;
				} else {
					inside = (w0 <= 0) && (w1 <= 0) && (w2 <= 0);
					if (w0 == 0.0f) inside = inside && is_start_edge0;
					if (w1 == 0.0f) inside = inside && is_start_edge1;
					if (w2 == 0.0f) inside = inside && is_start_edge2;
				}

				if (!inside) continue;

				float bary0 = w1 / area;
				float bary1 = w2 / area;
				float bary2 = w0 / area;
				float z = bary0 * a.z + bary1 * b.z + bary2 * c.z;

				Fragment frag;
				frag.fb_position.x = q.x;
				frag.fb_position.y = q.y;
				frag.fb_position.z = z;
				frag.attributes = va.attributes;

				for (auto& deriv : frag.derivatives) {
					deriv.x = 0.0f;
					deriv.y = 0.0f;
				}

				emit_fragment(frag);
			}
		}
	} else if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Smooth) {
		// A1T5: screen-space smooth triangles
		const auto& attr_a = va.attributes;
		const auto& attr_b = vb.attributes;
		const auto& attr_c = vc.attributes;
		size_t n_attr = attr_a.size();

		for (int32_t iy = min_iy; iy <= max_iy; ++iy) {
			for (int32_t ix = min_ix; ix <= max_ix; ++ix) {
				Vec2 q = Vec2(ix + 0.5f, iy + 0.5f);

				float w0 = (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x);
				float w1 = (c.x - b.x) * (q.y - b.y) - (c.y - b.y) * (q.x - b.x);
				float w2 = (a.x - c.x) * (q.y - c.y) - (a.y - c.y) * (q.x - c.x);

				bool inside;
				if (area > 0.0f) {
					inside = (w0 >= 0) && (w1 >= 0) && (w2 >= 0);
					if (w0 == 0.0f) inside = inside && is_start_edge0;
					if (w1 == 0.0f) inside = inside && is_start_edge1;
					if (w2 == 0.0f) inside = inside && is_start_edge2;
				} else {
					inside = (w0 <= 0) && (w1 <= 0) && (w2 <= 0);
					if (w0 == 0.0f) inside = inside && is_start_edge0;
					if (w1 == 0.0f) inside = inside && is_start_edge1;
					if (w2 == 0.0f) inside = inside && is_start_edge2;
				}

				if (!inside) continue;

				float bary0 = w1 / area;
				float bary1 = w2 / area;
				float bary2 = w0 / area;
				float z = bary0 * a.z + bary1 * b.z + bary2 * c.z;

				Fragment frag;
				frag.fb_position.x = q.x;
				frag.fb_position.y = q.y;
				frag.fb_position.z = z;

				// Interpolate attributes linearly in screen space
				for (size_t i = 0; i < n_attr; ++i) {
					frag.attributes[i] = attr_a[i] * bary0 + attr_b[i] * bary1 + attr_c[i] * bary2;
				}

				// Compute derivatives numerically using forward differences
				for (size_t i = 0; i < frag.derivatives.size(); ++i) {
					// d/dx
					Vec2 qx = Vec2(q.x + 1.0f, q.y);
					float w0x = (b.x - a.x) * (qx.y - a.y) - (b.y - a.y) * (qx.x - a.x);
					float w1x = (c.x - b.x) * (qx.y - b.y) - (c.y - b.y) * (qx.x - b.x);
					float w2x = (a.x - c.x) * (qx.y - c.y) - (a.y - c.y) * (qx.x - c.x);
					float bary0x = w1x / area;
					float bary1x = w2x / area;
					float bary2x = w0x / area;
					auto attr_x = attr_a[i] * bary0x + attr_b[i] * bary1x + attr_c[i] * bary2x;

					// d/dy
					Vec2 qy = Vec2(q.x, q.y + 1.0f);
					float w0y = (b.x - a.x) * (qy.y - a.y) - (b.y - a.y) * (qy.x - a.x);
					float w1y = (c.x - b.x) * (qy.y - b.y) - (c.y - b.y) * (qy.x - b.x);
					float w2y = (a.x - c.x) * (qy.y - c.y) - (a.y - c.y) * (qy.x - c.x);
					float bary0y = w1y / area;
					float bary1y = w2y / area;
					float bary2y = w0y / area;
					auto attr_y = attr_a[i] * bary0y + attr_b[i] * bary1y + attr_c[i] * bary2y;

					frag.derivatives[i].x = attr_x - frag.attributes[i];
					frag.derivatives[i].y = attr_y - frag.attributes[i];
				}

				emit_fragment(frag);
			}
		}
	} else if constexpr ((flags & PipelineMask_Interp) == Pipeline_Interp_Correct) {
		// A1T5: perspective correct triangles
		const auto& attr_a = va.attributes;
		const auto& attr_b = vb.attributes;
		const auto& attr_c = vc.attributes;
		size_t n_attr = attr_a.size();

		float inv_w_a = va.inv_w;
		float inv_w_b = vb.inv_w;
		float inv_w_c = vc.inv_w;

		for (int32_t iy = min_iy; iy <= max_iy; ++iy) {
			for (int32_t ix = min_ix; ix <= max_ix; ++ix) {
				Vec2 q = Vec2(ix + 0.5f, iy + 0.5f);

				float w0 = (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x);
				float w1 = (c.x - b.x) * (q.y - b.y) - (c.y - b.y) * (q.x - b.x);
				float w2 = (a.x - c.x) * (q.y - c.y) - (a.y - c.y) * (q.x - c.x);

				bool inside;
				if (area > 0.0f) {
					inside = (w0 >= 0) && (w1 >= 0) && (w2 >= 0);
					if (w0 == 0.0f) inside = inside && is_start_edge0;
					if (w1 == 0.0f) inside = inside && is_start_edge1;
					if (w2 == 0.0f) inside = inside && is_start_edge2;
				} else {
					inside = (w0 <= 0) && (w1 <= 0) && (w2 <= 0);
					if (w0 == 0.0f) inside = inside && is_start_edge0;
					if (w1 == 0.0f) inside = inside && is_start_edge1;
					if (w2 == 0.0f) inside = inside && is_start_edge2;
				}

				if (!inside) continue;

				float bary0 = w1 / area;
				float bary1 = w2 / area;
				float bary2 = w0 / area;
				float z = bary0 * a.z + bary1 * b.z + bary2 * c.z;

				// Perspective-correct barycentrics
				float wa = bary0 * inv_w_a;
				float wb = bary1 * inv_w_b;
				float wc = bary2 * inv_w_c;
				float sum_w = wa + wb + wc;
				if (sum_w == 0.0f) continue;
				wa /= sum_w;
				wb /= sum_w;
				wc /= sum_w;

				Fragment frag;
				frag.fb_position.x = q.x;
				frag.fb_position.y = q.y;
				frag.fb_position.z = z;

				for (size_t i = 0; i < n_attr; ++i) {
					frag.attributes[i] = attr_a[i] * wa + attr_b[i] * wb + attr_c[i] * wc;
				}

				// Compute derivatives numerically using forward differences
				for (size_t i = 0; i < frag.derivatives.size(); ++i) {
					// d/dx
					Vec2 qx = Vec2(q.x + 1.0f, q.y);
					float w0x = (b.x - a.x) * (qx.y - a.y) - (b.y - a.y) * (qx.x - a.x);
					float w1x = (c.x - b.x) * (qx.y - b.y) - (c.y - b.y) * (qx.x - b.x);
					float w2x = (a.x - c.x) * (qx.y - c.y) - (a.y - c.y) * (qx.x - c.x);
					float bary0x = w1x / area;
					float bary1x = w2x / area;
					float bary2x = w0x / area;
					float wa_x = bary0x * inv_w_a;
					float wb_x = bary1x * inv_w_b;
					float wc_x = bary2x * inv_w_c;
					float sum_wx = wa_x + wb_x + wc_x;
					if (sum_wx == 0.0f) sum_wx = 1.0f;
					wa_x /= sum_wx;
					wb_x /= sum_wx;
					wc_x /= sum_wx;
					auto attr_x = attr_a[i] * wa_x + attr_b[i] * wb_x + attr_c[i] * wc_x;

					// d/dy
					Vec2 qy = Vec2(q.x, q.y + 1.0f);
					float w0y = (b.x - a.x) * (qy.y - a.y) - (b.y - a.y) * (qy.x - a.x);
					float w1y = (c.x - b.x) * (qy.y - b.y) - (c.y - b.y) * (qy.x - b.x);
					float w2y = (a.x - c.x) * (qy.y - c.y) - (a.y - c.y) * (qy.x - c.x);
					float bary0y = w1y / area;
					float bary1y = w2y / area;
					float bary2y = w0y / area;
					float wa_y = bary0y * inv_w_a;
					float wb_y = bary1y * inv_w_b;
					float wc_y = bary2y * inv_w_c;
					float sum_wy = wa_y + wb_y + wc_y;
					if (sum_wy == 0.0f) sum_wy = 1.0f;
					wa_y /= sum_wy;
					wb_y /= sum_wy;
					wc_y /= sum_wy;
					auto attr_y = attr_a[i] * wa_y + attr_b[i] * wb_y + attr_c[i] * wc_y;

					frag.derivatives[i].x = attr_x - frag.attributes[i];
					frag.derivatives[i].y = attr_y - frag.attributes[i];
				}

				emit_fragment(frag);
			}
		}
	}
}

//-------------------------------------------------------------------------
// compile instantiations for all programs and blending and testing types:

#include "programs.h"

template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Always | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Always | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Never | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Never | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Always | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Always | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Never | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Never | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Less | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Less | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Less | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Always | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Always | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Never | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Never | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Less | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Less | Pipeline_Interp_Smooth>;
template struct Pipeline<PrimitiveType::Triangles, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Less | Pipeline_Interp_Correct>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Replace | Pipeline_Depth_Less | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Add | Pipeline_Depth_Less | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Always | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Never | Pipeline_Interp_Flat>;
template struct Pipeline<PrimitiveType::Lines, Programs::Lambertian,
                         Pipeline_Blend_Over | Pipeline_Depth_Less | Pipeline_Interp_Flat>;