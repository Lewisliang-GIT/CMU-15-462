
#include "bvh.h"
#include "aggregate.h"
#include "instance.h"
#include "tri_mesh.h"

#include <stack>

namespace PT {

struct BVHBuildData {
	BVHBuildData(size_t start, size_t range, size_t dst) : start(start), range(range), node(dst) {
	}
	size_t start; ///< start index into the primitive array
	size_t range; ///< range of index into the primitive array
	size_t node;  ///< address to update
};

struct SAHBucketData {
	BBox bb;          ///< bbox of all primitives
	size_t num_prims; ///< number of primitives in the bucket
};

template<typename Primitive>
void BVH<Primitive>::build(std::vector<Primitive>&& prims, size_t max_leaf_size) {
	//A3T3 - build a bvh

	// Keep these
	nodes.clear();
	primitives = std::move(prims);

	// Construct a BVH from the given vector of primitives and maximum leaf
    // size configuration.

	if (primitives.empty()) {
		root_idx = 0;
		return;
	}

	struct BuildEntry {
		size_t start, range, parent, child_idx;
	};

	// Compute bbox of a range of primitives
	auto compute_bbox = [&](size_t start, size_t range) {
		BBox box = primitives[start].bbox();
		for (size_t i = start + 1; i < start + range; i++) {
			box.enclose(primitives[i].bbox());
		}
		return box;
	};

	// Compute centroid bbox of a range of primitives
	auto compute_centroid_bbox = [&](size_t start, size_t range) {
		BBox box_c=primitives[start].bbox();
		for (size_t i = start + 1; i < start + range; i++) {
			box_c.enclose(primitives[i].bbox().center());
		}
		return box_c;
	};

	struct NodeBuild {
		size_t start, range;
		size_t parent, child_idx;
	};

	std::stack<NodeBuild> stack;
	root_idx = 0;
	stack.push({0, primitives.size(), size_t(-1), 0});

	while (!stack.empty()) {
		NodeBuild entry = stack.top();
		stack.pop();

		size_t start = entry.start;
		size_t range = entry.range;

		BBox node_bbox = compute_bbox(start, range);

		// Leaf node
		if (range <= max_leaf_size) {
			size_t node_idx = new_node(node_bbox, start, range, nodes.size(), nodes.size());
			if (entry.parent != size_t(-1)) {
				if (entry.child_idx == 0)
					nodes[entry.parent].l = node_idx;
				else
					nodes[entry.parent].r = node_idx;
			} else {
				root_idx = node_idx;
			}
			continue;
		}

		// Compute centroid bbox and choose split axis
		BBox centroid_bbox = compute_centroid_bbox(start, range);
		Vec3 extent = centroid_bbox.max - centroid_bbox.min;
		int axis = 0;
		if (extent.y > extent.x && extent.y > extent.z) axis = 1;
		else if (extent.z > extent.x && extent.z > extent.y) axis = 2;

		// If all centroids are the same, make a leaf
		if (extent[axis] < 1e-6f) {
			size_t node_idx = new_node(node_bbox, start, range, nodes.size(), nodes.size());
			if (entry.parent != size_t(-1)) {
				if (entry.child_idx == 0)
					nodes[entry.parent].l = node_idx;
				else
					nodes[entry.parent].r = node_idx;
			} else {
				root_idx = node_idx;
			}
			continue;
		}

		// Surface Area Heuristic (SAH) split
		const int n_buckets = 12;
		struct BucketInfo {
			BBox bbox;
			size_t count = 0;
		};
		BucketInfo buckets[n_buckets];

		// Assign primitives to buckets
		for (size_t i = start; i < start + range; i++) {
			Vec3 centroid = primitives[i].bbox().center();
			float offset = (centroid[axis] - centroid_bbox.min[axis]) / (extent[axis]);
			int b = int(n_buckets * offset);
			if (b == n_buckets) b = n_buckets - 1;
			b = std::clamp(b, 0, n_buckets - 1);
			buckets[b].count++;
			buckets[b].bbox.enclose(primitives[i].bbox());
		}

		// Test all possible splits
		float min_cost = std::numeric_limits<float>::infinity();
		int min_split = -1;
		float total_sa = node_bbox.surface_area();

		for (int i = 1; i < n_buckets; i++) {
			BBox b0, b1;
			size_t c0 = 0, c1 = 0;
			for (int j = 0; j < i; j++) {
				if (buckets[j].count) {
					b0.enclose(buckets[j].bbox);
					c0 += buckets[j].count;
				}
			}
			for (int j = i; j < n_buckets; j++) {
				if (buckets[j].count) {
					b1.enclose(buckets[j].bbox);
					c1 += buckets[j].count;
				}
			}
			if (c0 == 0 || c1 == 0) continue;
			float cost = 0.125f + (b0.surface_area() * c0 + b1.surface_area() * c1) / total_sa;
			if (cost < min_cost) {
				min_cost = cost;
				min_split = i;
			}
		}

		// If no good split, make a leaf
		if (min_split == -1) {
			size_t node_idx = new_node(node_bbox, start, range, nodes.size(), nodes.size());
			if (entry.parent != size_t(-1)) {
				if (entry.child_idx == 0)
					nodes[entry.parent].l = node_idx;
				else
					nodes[entry.parent].r = node_idx;
			} else {
				root_idx = node_idx;
			}
			continue;
		}

		// Partition primitives
		auto mid_iter = std::partition(
			primitives.begin() + start, primitives.begin() + start + range,
			[&](const Primitive& prim) {
				Vec3 centroid = prim.bbox().center();
				float offset = (centroid[axis] - centroid_bbox.min[axis]) / (extent[axis]);
				int b = int(n_buckets * offset);
				if (b == n_buckets) b = n_buckets - 1;
				b = std::clamp(b, 0, n_buckets - 1);
				return b < min_split;
			}
		);
		size_t mid = std::distance(primitives.begin(), mid_iter);

		// Create parent node
		size_t node_idx = new_node(node_bbox, start, range, 0, 0);
		if (entry.parent != size_t(-1)) {
			if (entry.child_idx == 0)
				nodes[entry.parent].l = node_idx;
			else
				nodes[entry.parent].r = node_idx;
		} else {
			root_idx = node_idx;
		}

		// Push children to stack
		stack.push({mid, start + range - mid, node_idx, 1});
		stack.push({start, mid - start, node_idx, 0});
	}
}

template<typename Primitive> Trace BVH<Primitive>::hit(const Ray& ray) const {
	// Efficient BVH traversal using an explicit stack

	if (nodes.empty()) return {};

	Trace closest;
	Vec2 times(ray.dist_bounds);

	std::stack<size_t> stack;
	stack.push(root_idx);

	while (!stack.empty()) {
		size_t idx = stack.top();
		stack.pop();

		const Node& node = nodes[idx];
		Vec2 node_times = times;
		if (!node.bbox.hit(ray, node_times)) continue;

		if (node.is_leaf()) {
			for (size_t i = node.start; i < node.start + node.size; i++) {
				Trace hit = primitives[i].hit(ray);
				closest = Trace::min(closest, hit);
			}
		} else {
			stack.push(node.l);
			stack.push(node.r);
		}
	}
	return closest;
}

template<typename Primitive>
BVH<Primitive>::BVH(std::vector<Primitive>&& prims, size_t max_leaf_size) {
	build(std::move(prims), max_leaf_size);
}

template<typename Primitive> std::vector<Primitive> BVH<Primitive>::destructure() {
	nodes.clear();
	return std::move(primitives);
}

template<typename Primitive>
template<typename P>
typename std::enable_if<std::is_copy_assignable_v<P>, BVH<P>>::type BVH<Primitive>::copy() const {
	BVH<Primitive> ret;
	ret.nodes = nodes;
	ret.primitives = primitives;
	ret.root_idx = root_idx;
	return ret;
}

template<typename Primitive> Vec3 BVH<Primitive>::sample(RNG &rng, Vec3 from) const {
	if (primitives.empty()) return {};
	int32_t n = rng.integer(0, static_cast<int32_t>(primitives.size()));
	return primitives[n].sample(rng, from);
}

template<typename Primitive>
float BVH<Primitive>::pdf(Ray ray, const Mat4& T, const Mat4& iT) const {
	if (primitives.empty()) return 0.0f;
	float ret = 0.0f;
	for (auto& prim : primitives) ret += prim.pdf(ray, T, iT);
	return ret / primitives.size();
}

template<typename Primitive> void BVH<Primitive>::clear() {
	nodes.clear();
	primitives.clear();
}

template<typename Primitive> bool BVH<Primitive>::Node::is_leaf() const {
	// A node is a leaf if l == r, since all interior nodes must have distinct children
	return l == r;
}

template<typename Primitive>
size_t BVH<Primitive>::new_node(BBox box, size_t start, size_t size, size_t l, size_t r) {
	Node n;
	n.bbox = box;
	n.start = start;
	n.size = size;
	n.l = l;
	n.r = r;
	nodes.push_back(n);
	return nodes.size() - 1;
}
 
template<typename Primitive> BBox BVH<Primitive>::bbox() const {
	if(nodes.empty()) return BBox{Vec3{0.0f}, Vec3{0.0f}};
	return nodes[root_idx].bbox;
}

template<typename Primitive> size_t BVH<Primitive>::n_primitives() const {
	return primitives.size();
}

template<typename Primitive>
uint32_t BVH<Primitive>::visualize(GL::Lines& lines, GL::Lines& active, uint32_t level,
                                   const Mat4& trans) const {

	std::stack<std::pair<size_t, uint32_t>> tstack;
	tstack.push({root_idx, 0u});
	uint32_t max_level = 0u;

	if (nodes.empty()) return max_level;

	while (!tstack.empty()) {

		auto [idx, lvl] = tstack.top();
		max_level = std::max(max_level, lvl);
		const Node& node = nodes[idx];
		tstack.pop();

		Spectrum color = lvl == level ? Spectrum(1.0f, 0.0f, 0.0f) : Spectrum(1.0f);
		GL::Lines& add = lvl == level ? active : lines;

		BBox box = node.bbox;
		box.transform(trans);
		Vec3 min = box.min, max = box.max;

		auto edge = [&](Vec3 a, Vec3 b) { add.add(a, b, color); };

		edge(min, Vec3{max.x, min.y, min.z});
		edge(min, Vec3{min.x, max.y, min.z});
		edge(min, Vec3{min.x, min.y, max.z});
		edge(max, Vec3{min.x, max.y, max.z});
		edge(max, Vec3{max.x, min.y, max.z});
		edge(max, Vec3{max.x, max.y, min.z});
		edge(Vec3{min.x, max.y, min.z}, Vec3{max.x, max.y, min.z});
		edge(Vec3{min.x, max.y, min.z}, Vec3{min.x, max.y, max.z});
		edge(Vec3{min.x, min.y, max.z}, Vec3{max.x, min.y, max.z});
		edge(Vec3{min.x, min.y, max.z}, Vec3{min.x, max.y, max.z});
		edge(Vec3{max.x, min.y, min.z}, Vec3{max.x, max.y, min.z});
		edge(Vec3{max.x, min.y, min.z}, Vec3{max.x, min.y, max.z});

		if (!node.is_leaf()) {
			tstack.push({node.l, lvl + 1});
			tstack.push({node.r, lvl + 1});
		} else {
			for (size_t i = node.start; i < node.start + node.size; i++) {
				uint32_t c = primitives[i].visualize(lines, active, level - lvl, trans);
				max_level = std::max(c + lvl, max_level);
			}
		}
	}
	return max_level;
}

template class BVH<Triangle>;
template class BVH<Instance>;
template class BVH<Aggregate>;
template BVH<Triangle> BVH<Triangle>::copy<Triangle>() const;

} // namespace PT
