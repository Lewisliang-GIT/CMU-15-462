#include "halfedge.h"

#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <iostream>

/******************************************************************
*********************** Local Operations **************************
******************************************************************/

/* Note on local operation return types:

    The local operations all return a std::optional<T> type. This is used so that your
    implementation can signify that it cannot perform an operation (i.e., because
    the resulting mesh does not have a valid representation).

    An optional can have two values: std::nullopt, or a value of the type it is
    parameterized on. In this way, it's similar to a pointer, but has two advantages:
    the value it holds need not be allocated elsewhere, and it provides an API that
    forces the user to check if it is null before using the value.

    In your implementation, if you have successfully performed the operation, you can
    simply return the required reference:

            ... collapse the edge ...
            return collapsed_vertex_ref;

    And if you wish to deny the operation, you can return the null optional:

            return std::nullopt;

    Note that the stubs below all reject their duties by returning the null optional.
*/


/*
 * add_face: add a standalone face to the mesh
 *  sides: number of sides
 *  radius: distance from vertices to origin
 *
 * We provide this method as an example of how to make new halfedge mesh geometry.
 */
std::optional<Halfedge_Mesh::FaceRef> Halfedge_Mesh::add_face(uint32_t sides, float radius) {
	//faces with fewer than three sides are invalid, so abort the operation:
	if (sides < 3) return std::nullopt;


	std::vector< VertexRef > face_vertices;
	//In order to make the first edge point in the +x direction, first vertex should
	// be at -90.0f - 0.5f * 360.0f / float(sides) degrees, so:
	float const start_angle = (-0.25f - 0.5f / float(sides)) * 2.0f * PI_F;
	for (uint32_t s = 0; s < sides; ++s) {
		float angle = float(s) / float(sides) * 2.0f * PI_F + start_angle;
		VertexRef v = emplace_vertex();
		v->position = radius * Vec3(std::cos(angle), std::sin(angle), 0.0f);
		face_vertices.emplace_back(v);
	}

	assert(face_vertices.size() == sides);

	//assemble the rest of the mesh parts:
	FaceRef face = emplace_face(false); //the face to return
	FaceRef boundary = emplace_face(true); //the boundary loop around the face

	std::vector< HalfedgeRef > face_halfedges; //will use later to set ->next pointers

	for (uint32_t s = 0; s < sides; ++s) {
		//will create elements for edge from a->b:
		VertexRef a = face_vertices[s];
		VertexRef b = face_vertices[(s+1)%sides];

		//h is the edge on face:
		HalfedgeRef h = emplace_halfedge();
		//t is the twin, lies on boundary:
		HalfedgeRef t = emplace_halfedge();
		//e is the edge corresponding to h,t:
		EdgeRef e = emplace_edge(false); //false: non-sharp

		//set element data to something reasonable:
		//(most ops will do this with interpolate_data(), but no data to interpolate here)
		h->corner_uv = a->position.xy() / (2.0f * radius) + 0.5f;
		h->corner_normal = Vec3(0.0f, 0.0f, 1.0f);
		t->corner_uv = b->position.xy() / (2.0f * radius) + 0.5f;
		t->corner_normal = Vec3(0.0f, 0.0f,-1.0f);

		//thing -> halfedge pointers:
		e->halfedge = h;
		a->halfedge = h;
		if (s == 0) face->halfedge = h;
		if (s + 1 == sides) boundary->halfedge = t;

		//halfedge -> thing pointers (except 'next' -- will set that later)
		h->twin = t;
		h->vertex = a;
		h->edge = e;
		h->face = face;

		t->twin = h;
		t->vertex = b;
		t->edge = e;
		t->face = boundary;

		face_halfedges.emplace_back(h);
	}

	assert(face_halfedges.size() == sides);

	for (uint32_t s = 0; s < sides; ++s) {
		face_halfedges[s]->next = face_halfedges[(s+1)%sides];
		face_halfedges[(s+1)%sides]->twin->next = face_halfedges[s]->twin;
	}

	return face;
}


/*
 * bisect_edge: split an edge without splitting the adjacent faces
 *  e: edge to split
 *
 * returns: added vertex
 *
 * We provide this as an example for how to implement local operations.
 * (and as a useful subroutine!)
 */
std::optional<Halfedge_Mesh::VertexRef> Halfedge_Mesh::bisect_edge(EdgeRef e) {
	// Phase 0: draw a picture
	//
	// before:
	//    ----h--->
	// v1 ----e--- v2
	//   <----t---
	//
	// after:
	//    --h->    --h2->
	// v1 --e-- vm --e2-- v2
	//    <-t2-    <--t--
	//

	// Phase 1: collect existing elements
	HalfedgeRef h = e->halfedge;
	HalfedgeRef t = h->twin;
	VertexRef v1 = h->vertex;
	VertexRef v2 = t->vertex;

	// Phase 2: Allocate new elements, set data
	VertexRef vm = emplace_vertex();
	vm->position = (v1->position + v2->position) / 2.0f;
	interpolate_data({v1, v2}, vm); //set bone_weights

	EdgeRef e2 = emplace_edge();
	e2->sharp = e->sharp; //copy sharpness flag

	HalfedgeRef h2 = emplace_halfedge();
	interpolate_data({h, h->next}, h2); //set corner_uv, corner_normal

	HalfedgeRef t2 = emplace_halfedge();
	interpolate_data({t, t->next}, t2); //set corner_uv, corner_normal

	// The following elements aren't necessary for the bisect_edge, but they are here to demonstrate phase 4
    FaceRef f_not_used = emplace_face();
    HalfedgeRef h_not_used = emplace_halfedge();

	// Phase 3: Reassign connectivity (careful about ordering so you don't overwrite values you may need later!)

	vm->halfedge = h2;

	e2->halfedge = h2;

	assert(e->halfedge == h); //unchanged

	//n.b. h remains on the same face so even if h->face->halfedge == h, no fixup needed (t, similarly)

	h2->twin = t;
	h2->next = h->next;
	h2->vertex = vm;
	h2->edge = e2;
	h2->face = h->face;

	t2->twin = h;
	t2->next = t->next;
	t2->vertex = vm;
	t2->edge = e;
	t2->face = t->face;
	
	h->twin = t2;
	h->next = h2;
	assert(h->vertex == v1); // unchanged
	assert(h->edge == e); // unchanged
	//h->face unchanged

	t->twin = h2;
	t->next = t2;
	assert(t->vertex == v2); // unchanged
	t->edge = e2;
	//t->face unchanged


	// Phase 4: Delete unused elements
    erase_face(f_not_used);
    erase_halfedge(h_not_used);

	// Phase 5: Return the correct iterator
	return vm;
}


/*
 * split_edge: split an edge and adjacent (non-boundary) faces
 *  e: edge to split
 *
 * returns: added vertex. vertex->halfedge should lie along e
 *
 * Note that when splitting the adjacent faces, the new edge
 * should connect to the vertex ccw from the ccw-most end of e
 * within the face.
 *
 * Do not split adjacent boundary faces.
 */
std::optional<Halfedge_Mesh::VertexRef> Halfedge_Mesh::split_edge(EdgeRef e) {
	// A2L2 (REQUIRED): split_edge
	
	auto optional_v = bisect_edge(e);
	if (!optional_v) return std::nullopt;
	VertexRef v = *optional_v;

	// Check if both adjacent faces are not boundary
	if (!(e->halfedge->face->boundary || e->halfedge->twin->face->boundary)) {
		// Get halfedges around the new vertex
		HalfedgeRef h = v->halfedge->twin->next->twin;
		HalfedgeRef t = v->halfedge->twin;
		HalfedgeRef h_next2 = h->next->next->next;
		HalfedgeRef t_next2 = t->next->next->next;

		// Create new elements
		EdgeRef e1 = emplace_edge(), e2 = emplace_edge();
		HalfedgeRef h1 = emplace_halfedge(), t1 = emplace_halfedge();
		HalfedgeRef h2 = emplace_halfedge(), t2 = emplace_halfedge();
		FaceRef f1 = emplace_face(), f2 = emplace_face();

		// Setup twins, vertices, edges
		h1->twin = t1; t1->twin = h1;
		h2->twin = t2; t2->twin = h2;
		h1->vertex = h2->vertex = v;
		t1->vertex = h_next2->vertex;
		t2->vertex = t_next2->vertex;
		e1->halfedge = h1; e2->halfedge = h2;
		h1->edge = t1->edge = e1;
		h2->edge = t2->edge = e2;

		// Setup next pointers
		h1->next = h_next2; t1->next = h->next;
		h2->next = t_next2; t2->next = h->twin;
		h->next->next->next = t1;
		t->next->next->next = t2;
		h->next = h1; t->next = h2;

		// Setup faces
		f1->boundary = f2->boundary = false;
		f1->halfedge = t1; f2->halfedge = t2;
		for (HalfedgeRef he = t1; ; he = he->next) {
			he->face = f1;
			if (he->next == t1) break;
		}
		for (HalfedgeRef he = t2; ; he = he->next) {
			he->face = f2;
			if (he->next == t2) break;
		}
		h1->face = h->face; h2->face = t->face;
		h->face->halfedge = h; t->face->halfedge = t;

		// Interpolate data
		interpolate_data({h, h1->next}, h1);
		interpolate_data({t1, t1->next}, t1);
		interpolate_data({t, h2->next}, h2);
		interpolate_data({t2, t2->next}, t2);

		return v;
	} else {
		// Only one adjacent face is non-boundary
		HalfedgeRef h = v->halfedge->face->boundary ? v->halfedge->next->twin : v->halfedge->twin->next->twin;
		HalfedgeRef h_next2 = h->next->next->next;

		EdgeRef e1 = emplace_edge();
		HalfedgeRef h1 = emplace_halfedge(), t1 = emplace_halfedge();
		FaceRef f1 = emplace_face();

		h1->twin = t1; t1->twin = h1;
		h1->vertex = v; t1->vertex = h_next2->vertex;
		e1->halfedge = h1; h1->edge = t1->edge = e1;

		h1->next = h_next2; t1->next = h->next;
		h->next->next->next = t1;
		h->next = h1;

		f1->boundary = false; f1->halfedge = t1;
		for (HalfedgeRef he = t1; ; he = he->next) {
			he->face = f1;
			if (he->next == t1) break;
		}
		h1->face = h->face;
		h->face->halfedge = h;

		interpolate_data({h, h1->next}, h1);
		interpolate_data({t1, t1->next}, t1);

		return v;
	}
}



/*
 * inset_vertex: divide a face into triangles by placing a vertex at f->center()
 *  f: the face to add the vertex to
 *
 * returns:
 *  std::nullopt if insetting a vertex would make mesh invalid
 *  the inset vertex otherwise
 */
std::optional<Halfedge_Mesh::VertexRef> Halfedge_Mesh::inset_vertex(FaceRef f) {
	// A2Lx4 (OPTIONAL): inset vertex
	
	(void)f;
    return std::nullopt;
}


/* [BEVEL NOTE] Note on the beveling process:

	Each of the bevel_vertex, bevel_edge, and extrude_face functions do not represent
	a full bevel/extrude operation. Instead, they should update the _connectivity_ of
	the mesh, _not_ the positions of newly created vertices. In fact, you should set
	the positions of new vertices to be exactly the same as wherever they "started from."

	When you click on a mesh element while in bevel mode, one of those three functions
	is called. But, because you may then adjust the distance/offset of the newly
	beveled face, we need another method of updating the positions of the new vertices.

	This is where bevel_positions and extrude_positions come in: these functions are
	called repeatedly as you move your mouse, the position of which determines the
	amount / shrink parameters. These functions are also passed an array of the original
	vertex positions, stored just after the bevel/extrude call, in order starting at
	face->halfedge->vertex, and the original element normal, computed just *before* the
	bevel/extrude call.

	Finally, note that the amount, extrude, and/or shrink parameters are not relative
	values -- you should compute a particular new position from them, not a delta to
	apply.
*/

/*
 * bevel_vertex: creates a face in place of a vertex
 *  v: the vertex to bevel
 *
 * returns: reference to the new face
 *
 * see also [BEVEL NOTE] above.
 */
std::optional<Halfedge_Mesh::FaceRef> Halfedge_Mesh::bevel_vertex(VertexRef v) {
	//A2Lx5 (OPTIONAL): Bevel Vertex
	// Reminder: This function does not update the vertex positions.
	// Remember to also fill in bevel_vertex_helper (A2Lx5h)

	(void)v;
    return std::nullopt;
}

/*
 * bevel_edge: creates a face in place of an edge
 *  e: the edge to bevel
 *
 * returns: reference to the new face
 *
 * see also [BEVEL NOTE] above.
 */
std::optional<Halfedge_Mesh::FaceRef> Halfedge_Mesh::bevel_edge(EdgeRef e) {
	//A2Lx6 (OPTIONAL): Bevel Edge
	// Reminder: This function does not update the vertex positions.
	// remember to also fill in bevel_edge_helper (A2Lx6h)

	(void)e;
    return std::nullopt;
}

/*
 * extrude_face: creates a face inset into a face
 *  f: the face to inset
 *
 * returns: reference to the inner face
 *
 * see also [BEVEL NOTE] above.
 */
std::optional<Halfedge_Mesh::FaceRef> Halfedge_Mesh::extrude_face(FaceRef f) {
    //A2L4: Extrude Face
    // Reminder: This function does not update the vertex positions.
    // Remember to also fill in extrude_helper (A2L4h)
   	FaceRef f_boundary = f->halfedge->face;
	if (!f_boundary->boundary) {
		f_boundary = f->halfedge->twin->face;
	}

	HalfedgeRef h = f->halfedge;
	HalfedgeRef t = h->twin;

	std::cout << h->id << " " << t->id << std::endl;

	// create new points for the face
	std::vector<VertexRef> old_vertices;
	std::vector<VertexRef> new_vertices;
	HalfedgeRef temp = h;
	do {
		VertexRef v = emplace_vertex();
		v->position = temp->vertex->position;
		old_vertices.push_back(temp->vertex);
		new_vertices.push_back(v);
		temp = temp->next;
	} while (temp != h);

	// create new halfedges and edges for the face itself
	for (size_t i = 0; i < new_vertices.size(); i++) {
		HalfedgeRef h_i = emplace_halfedge();
		HalfedgeRef t_i = emplace_halfedge();
		EdgeRef e = emplace_edge();
		e->halfedge = h_i;
		h_i->edge = e;
		t_i->edge = e;
		h_i->twin = t_i;
		t_i->twin = h_i;
		h_i->vertex = new_vertices[i];
		t_i->vertex = new_vertices[(i + 1) % new_vertices.size()];
		new_vertices[i]->halfedge = h_i;
		h_i->face = f;
		t_i->face = f_boundary;
	}

	std::vector<HalfedgeRef> outer_edges;
	// connect the new halfedges to each other
	for (size_t i = 0; i < new_vertices.size(); i++) {
		HalfedgeRef h_i = new_vertices[i]->halfedge;
		h_i->next = new_vertices[(i + 1) % new_vertices.size()]->halfedge;
		outer_edges.push_back(h_i->twin);
	}

	f->halfedge = new_vertices[0]->halfedge;

	std::vector<HalfedgeRef> outer_joints;
	std::vector<HalfedgeRef> inner_joints;

	// connect all the vertices from the new face to the old face with halfedges
	for (size_t i = 0; i < new_vertices.size(); i++) {
		EdgeRef e = emplace_edge();
		HalfedgeRef h_i = emplace_halfedge();
		HalfedgeRef t_i = emplace_halfedge();
		e->halfedge = h_i;
		h_i->edge = e;
		t_i->edge = e;
		h_i->twin = t_i;
		t_i->twin = h_i;
		h_i->vertex = old_vertices[i];
		t_i->vertex = new_vertices[i];
		h_i->face = f_boundary;
		t_i->face = f_boundary;
		outer_joints.push_back(h_i);
		inner_joints.push_back(t_i);
	}

	// set next pointers for the new halfedges
	for (size_t i = 0; i < new_vertices.size(); i++) {
		outer_joints[i]->next = outer_edges[(i - 1) % new_vertices.size()];
		outer_edges[i]->next = inner_joints[i];
		inner_joints[i]->next = old_vertices[i]->halfedge;
		old_vertices[i]->halfedge->next = outer_joints[(i + 1) % new_vertices.size()];
	}

	// create new faces with the newly connected halfedges
	for (size_t i = 0; i < new_vertices.size(); i++) {
		FaceRef f_ = emplace_face();
		f_->halfedge = outer_joints[i];
		inner_joints[(i - 1) % new_vertices.size()]->face = f_;
		old_vertices[(i - 1) % new_vertices.size()]->halfedge->face = f_;
		outer_edges[(i - 1) % new_vertices.size()]->face = f_;
		outer_joints[i]->face = f_;
	}

	return f;
}

/*
 * flip_edge: rotate non-boundary edge ccw inside its containing faces
 *  e: edge to flip
 *
 * if e is a boundary edge, does nothing and returns std::nullopt
 * if flipping e would create an invalid mesh, does nothing and returns std::nullopt
 *
 * otherwise returns the edge, post-rotation
 *
 * does not create or destroy mesh elements.
 */
std::optional<Halfedge_Mesh::EdgeRef> Halfedge_Mesh::flip_edge(EdgeRef e) {
	//A2L1: Flip Edge
	if (e->on_boundary()) {
		return std::nullopt;
	}

	HalfedgeRef& h = e->halfedge;
	HalfedgeRef& t = h->twin;
	
	// update the vertex's halfedge
	h->vertex->halfedge = h->vertex->halfedge == h ? h->twin->next : h->vertex->halfedge;
	t->vertex->halfedge = t->vertex->halfedge == t ? t->twin->next : t->vertex->halfedge;
	
	// update the h and t
	HalfedgeRef h_next = h->next;
	HalfedgeRef t_next = t->next;
	HalfedgeRef h_next2 = h->next->next;
	HalfedgeRef t_next2 = t->next->next;
	HalfedgeRef h_last = h;
	HalfedgeRef t_last = t;
	while (h_last->next != h) h_last = h_last->next;
	while (t_last->next != t) t_last = t_last->next;

	// make sure face's halfedge is still valid after flip
	h->face->halfedge = h;
	t->face->halfedge = t;

	// update the new start vertex
	h->vertex = t_next2->vertex;
	t->vertex = h_next2->vertex;
	
	// update the face and nextabout the changed vertex
	h_next->face = t->face;
	t_next->face = h->face;
	h_next->next = t;
	t_next->next = h;
	h_last->next = t_next;
	t_last->next = h_next;

	// update the h and t
	h->next = h_next2;
	t->next = t_next2;
	return e;
}


/*
 * make_boundary: add non-boundary face to boundary
 *  face: the face to make part of the boundary
 *
 * if face ends up adjacent to other boundary faces, merge them into face
 *
 * if resulting mesh would be invalid, does nothing and returns std::nullopt
 * otherwise returns face
 */
std::optional<Halfedge_Mesh::FaceRef> Halfedge_Mesh::make_boundary(FaceRef face) {
	//A2Lx7: (OPTIONAL) make_boundary

	return std::nullopt; //TODO: actually write this code!
}

/*
 * dissolve_vertex: merge non-boundary faces adjacent to vertex, removing vertex
 *  v: vertex to merge around
 *
 * if merging would result in an invalid mesh, does nothing and returns std::nullopt
 * otherwise returns the merged face
 */
std::optional<Halfedge_Mesh::FaceRef> Halfedge_Mesh::dissolve_vertex(VertexRef v) {
	// A2Lx1 (OPTIONAL): Dissolve Vertex

    return std::nullopt;
}

/*
 * dissolve_edge: merge the two faces on either side of an edge
 *  e: the edge to dissolve
 *
 * merging a boundary and non-boundary face produces a boundary face.
 *
 * if the result of the merge would be an invalid mesh, does nothing and returns std::nullopt
 * otherwise returns the merged face.
 */
std::optional<Halfedge_Mesh::FaceRef> Halfedge_Mesh::dissolve_edge(EdgeRef e) {
	// A2Lx2 (OPTIONAL): dissolve_edge

	//Reminder: use interpolate_data() to merge corner_uv / corner_normal data
	
    return std::nullopt;
}

/* collapse_edge: collapse edge to a vertex at its middle
 *  e: the edge to collapse
 *
 * if collapsing the edge would result in an invalid mesh, does nothing and returns std::nullopt
 * otherwise returns the newly collapsed vertex
 */
std::optional<Halfedge_Mesh::VertexRef> Halfedge_Mesh::collapse_edge(EdgeRef e) {
	//A2L3: Collapse Edge

	//Reminder: use interpolate_data() to merge corner_uv / corner_normal data on halfedges
	// (also works for bone_weights data on vertices!)

    HalfedgeRef h = e->halfedge;
    HalfedgeRef t = h->twin;
    VertexRef v1 = h->vertex;
    VertexRef v2 = t->vertex;
    VertexRef new_v = emplace_vertex();
    new_v->position = (v1->position + v2->position) / 2.0f;
    interpolate_data({v1, v2}, new_v);
    // corner case exist, need to modify
    new_v->halfedge = t->next;
    HalfedgeRef h_pre, t_pre;
    HalfedgeRef tmp = h;
    while(tmp->next != h) 
        tmp = tmp->next;
    h_pre = tmp;
    tmp = t;
    while(tmp->next != t)
        tmp = tmp->next;
    t_pre = tmp;
    // get all adjcent edge on v1 and v2
    std::vector<HalfedgeRef> halfedge_on_v1 = {};
    std::vector<HalfedgeRef> halfedge_on_v2 = {};
    tmp = h;
    while(tmp->twin->next != h) {
        tmp = tmp->twin->next;
        halfedge_on_v1.push_back(tmp);
    }
    tmp = t;
    while(tmp->twin->next != t) {
        tmp = tmp->twin->next;
        halfedge_on_v2.push_back(tmp);
    }
    for(auto halfedge : halfedge_on_v1) 
        halfedge->vertex = new_v;
    for(auto halfedge: halfedge_on_v2)
        halfedge->vertex = new_v;
    if(h->next->next->next == h) {
        FaceRef f = h->face;
        erase_face(f);
        erase_edge(h->next->edge);
        h->next->next->twin->twin = h->next->twin;
        h->next->twin->twin = h->next->next->twin;
        h->next->twin->edge = h->next->next->edge;
        new_v->halfedge = h->next->twin;
        h->next->next->vertex->halfedge = h->next->next->twin->next;
        h->next->next->edge->halfedge = h->next->next->twin;
        h->next->edge->halfedge = h->next->twin;
        erase_halfedge(h->next->next);
        erase_halfedge(h->next);
    }else {
        h_pre->next = h->next;
        h->face->halfedge = h->next;
        new_v->halfedge = h_pre->twin;
    }
    if(t->next->next->next == t) {
        FaceRef f = t->face;
        erase_face(f);
        erase_edge(t->next->edge);
        t->next->next->twin->twin = t->next->twin;
        t->next->twin->twin = t->next->next->twin;
        t->next->twin->edge = t->next->next->edge;
        new_v->halfedge = t->next->twin;
        t->next->next->vertex->halfedge = t->next->next->twin->next;
        t->next->next->edge->halfedge = t->next->next->twin;
        t->next->edge->halfedge = h->next->twin;
        erase_halfedge(t->next->next);
        erase_halfedge(t->next);
    }else {
        t_pre->next = t->next;
        t->face->halfedge = t->next;
        new_v->halfedge = t_pre->twin;
    }
    erase_edge(h->edge);
    erase_halfedge(h);
    erase_halfedge(t);
    erase_vertex(v1);
    erase_vertex(v2);
    
    return new_v;
}

/*
 * collapse_face: collapse a face to a single vertex at its center
 *  f: the face to collapse
 *
 * if collapsing the face would result in an invalid mesh, does nothing and returns std::nullopt
 * otherwise returns the newly collapsed vertex
 */
std::optional<Halfedge_Mesh::VertexRef> Halfedge_Mesh::collapse_face(FaceRef f) {
	//A2Lx3 (OPTIONAL): Collapse Face

	//Reminder: use interpolate_data() to merge corner_uv / corner_normal data on halfedges
	// (also works for bone_weights data on vertices!)

    return std::nullopt;
}

/*
 * weld_edges: glue two boundary edges together to make one non-boundary edge
 *  e, e2: the edges to weld
 *
 * if welding the edges would result in an invalid mesh, does nothing and returns std::nullopt
 * otherwise returns e, updated to represent the newly-welded edge
 */
std::optional<Halfedge_Mesh::EdgeRef> Halfedge_Mesh::weld_edges(EdgeRef e, EdgeRef e2) {
	//A2Lx8: Weld Edges

	//Reminder: use interpolate_data() to merge bone_weights data on vertices!

    return std::nullopt;
}



/*
 * bevel_positions: compute new positions for the vertices of a beveled vertex/edge
 *  face: the face that was created by the bevel operation
 *  start_positions: the starting positions of the vertices
 *     start_positions[i] is the starting position of face->halfedge(->next)^i
 *  direction: direction to bevel in (unit vector)
 *  distance: how far to bevel
 *
 * push each vertex from its starting position along its outgoing edge until it has
 *  moved distance `distance` in direction `direction`. If it runs out of edge to
 *  move along, you may choose to extrapolate, clamp the distance, or do something
 *  else reasonable.
 *
 * only changes vertex positions (no connectivity changes!)
 *
 * This is called repeatedly as the user interacts, just after bevel_vertex or bevel_edge.
 * (So you can assume the local topology is set up however your bevel_* functions do it.)
 *
 * see also [BEVEL NOTE] above.
 */
void Halfedge_Mesh::bevel_positions(FaceRef face, std::vector<Vec3> const &start_positions, Vec3 direction, float distance) {
	//A2Lx5h / A2Lx6h (OPTIONAL): Bevel Positions Helper
	
	// The basic strategy here is to loop over the list of outgoing halfedges,
	// and use the preceding and next vertex position from the original mesh
	// (in the start_positions array) to compute an new vertex position.
	
}

/*
 * extrude_positions: compute new positions for the vertices of an extruded face
 *  face: the face that was created by the extrude operation
 *  move: how much to translate the face
 *  shrink: amount to linearly interpolate vertices in the face toward the face's centroid
 *    shrink of zero leaves the face where it is
 *    positive shrink makes the face smaller (at shrink of 1, face is a point)
 *    negative shrink makes the face larger
 *
 * only changes vertex positions (no connectivity changes!)
 *
 * This is called repeatedly as the user interacts, just after extrude_face.
 * (So you can assume the local topology is set up however your extrude_face function does it.)
 *
 * Using extrude face in the GUI will assume a shrink of 0 to only extrude the selected face
 * Using bevel face in the GUI will allow you to shrink and increase the size of the selected face
 * 
 * see also [BEVEL NOTE] above.
 */
void Halfedge_Mesh::extrude_positions(FaceRef face, Vec3 move, float shrink) {
    //A2L4h: Extrude Positions Helper
    // Collect all vertices of the face
    std::vector<VertexRef> verts;
    HalfedgeRef h = face->halfedge;
    HalfedgeRef start = h;
    while (h != start->next) {
        verts.push_back(start->vertex);
        start = start->next;
    } 
	verts.push_back(start->vertex); // Add the last vertex
    size_t n = verts.size();
    // Compute centroid
    Vec3 centroid(0.0f,0.0f,0.0f);
    for (VertexRef v : verts) centroid += v->position;
    centroid /= float(n);
    // Move and shrink
    for (VertexRef v : verts) {
        v->position = (1.0f - shrink) * v->position + shrink * centroid + move;
    }
}

