#include "bvh.h"

static inline float getMin(float x, float y, float z) {
	return fminf(fminf(x, y), z);
}

static inline float getMax(float x, float y, float z) {
	return fmaxf(fmaxf(x, y), z);
}

AABB boundingBox(const Geom& geom) {
	AABB box;
	if (geom.type == TRIANGLE) {
		glm::vec4 v0 = geom.transform * glm::vec4(geom.vertices[0], 1.0f);
		glm::vec4 v1 = geom.transform * glm::vec4(geom.vertices[1], 1.0f);
		glm::vec4 v2 = geom.transform * glm::vec4(geom.vertices[2], 1.0f);

		box.min = glm::vec3(
			getMin(v0.x, v1.x, v2.x),
			getMin(v0.y, v1.y, v2.y),
			getMin(v0.z, v1.z, v2.z)
		);
		box.max = glm::vec3(
			getMax(v0.x, v1.x, v2.x),
			getMax(v0.y, v1.y, v2.y),
			getMax(v0.z, v1.z, v2.z)
		);
	}
	else {
		glm::vec3 mins(FLT_MAX), maxs(-FLT_MAX);

		glm::vec4 corners[8] = {
			geom.transform * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f),
			geom.transform * glm::vec4(-0.5f, -0.5f,  0.5f, 1.0f),
			geom.transform * glm::vec4(-0.5f,  0.5f, -0.5f, 1.0f),
			geom.transform * glm::vec4(0.5f, -0.5f, -0.5f, 1.0f),
			geom.transform * glm::vec4(-0.5f,  0.5f,  0.5f, 1.0f),
			geom.transform * glm::vec4(0.5f, -0.5f,  0.5f, 1.0f),
			geom.transform * glm::vec4(0.5f,  0.5f, -0.5f, 1.0f),
			geom.transform * glm::vec4(0.5f,  0.5f,  0.5f, 1.0f),
		};

		for (int i = 0; i < 8; i++) {
			mins = glm::min(mins, glm::vec3(corners[i]));
			maxs = glm::max(maxs, glm::vec3(corners[i]));
		}

		box.min = mins;
		box.max = maxs;
	}
	return box;
}


void BVH::IntersectBVH(Ray& ray, const std::vector<Geom>& geoms, int nodeIdx, ShadeableIntersection& isect)
{
	BVHNode& node = bvhNode[nodeIdx];

	// test bounding box
	if (!bboxIntersectionTest(ray, node.bboxMin, node.bboxMax))
		return;

	if (node.isLeaf())
	{
		for (int i = 0; i < node.count; ++i) {
			int geomIdx = orderedGeomIndices[node.start + i];
			glm::vec3 intersectionPoint, normal;
			bool wo = true;

			float t = -1;
			if (geoms[geomIdx].type == TRIANGLE) {
				t = triangleIntersectionTest(ray, geoms[geomIdx], intersectionPoint, normal, wo);
			}
			else if (geoms[geomIdx].type == SPHERE) {
				t = sphereIntersectionTest(geoms[geomIdx], ray, intersectionPoint, normal, wo);
			}
			else if (geoms[geomIdx].type == CUBE) {
				t = boxIntersectionTest(geoms[geomIdx], ray, intersectionPoint, normal, wo);
			}
			if (t > 0 && t < ray.t) {
				ray.t = t;
				isect.t = t;
				isect.surfaceNormal = normal;
				isect.materialId = geoms[geomIdx].materialid;
			}
		}
	}
	else
	{
		IntersectBVH(ray, geoms, node.left, isect);
		IntersectBVH(ray, geoms, node.right, isect);
	}
}


void BVH::build(const std::vector<Geom>& geoms) {
	int N = geoms.size();
	int maxNodes = 2 * N - 1;

	bvhNode.clear();
	bvhNode.resize(maxNodes);
	nodesUsed = 1;
	rootNodeIdx = 0;

	orderedGeomIndices.resize(geoms.size());
	for (int k = 0; k < geoms.size(); ++k)
		orderedGeomIndices[k] = k;

	bvhNode[rootNodeIdx].start = 0;
	bvhNode[rootNodeIdx].count = N;

	UpdateNodeBounds(rootNodeIdx, geoms); // makes the bounding box of the root node
	Subdivide(rootNodeIdx, geoms); // recursively subdivides the root node to build out the tree
}

void BVH::UpdateNodeBounds(int idx, const std::vector<Geom>& geoms) {
	BVHNode& node = bvhNode[idx];

	if (node.count <= 0) {
		node.bboxMin = glm::vec3(FLT_MAX);
		node.bboxMax = glm::vec3(-FLT_MAX);
		return;
	}

	node.bboxMin = glm::vec3(FLT_MAX);
	node.bboxMax = glm::vec3(-FLT_MAX);

	int Nidx = (int)orderedGeomIndices.size();
	if (node.start < 0 || node.start > Nidx) return;
	if (node.start + node.count > Nidx) {
		// invalid assignment — clamp to available range (diagnostic)
		node.count = std::max(0, Nidx - node.start);
	}

	for (int i = 0; i < node.count; i++) { // loop thru all prims
		int geomIdx = orderedGeomIndices[node.start + i];   // index of the primitive
		const Geom& g = geoms[geomIdx];

		auto box = boundingBox(g);
		node.bboxMin = glm::min(node.bboxMin, box.min);
		node.bboxMax = glm::max(node.bboxMax, box.max);
	}
}

void BVH::Subdivide(int idx, const std::vector<Geom>& geoms) {
	BVHNode& node = bvhNode[idx];

	if (node.count <= 2) return; // kill recursion if too small

	glm::vec3 extent = node.bboxMax - node.bboxMin;

	// get axis with largest extent to split along
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;
	float splitPos = node.bboxMin[axis] + extent[axis] * 0.5f; // activate split at the middle of whatever axis we chose previously

	// partitioning...
	int i = node.start;
	int j = i + node.count - 1;

	while (i <= j) {
		int geomIdx = orderedGeomIndices[i];
		glm::vec3 centroid = geoms[geomIdx].centroid();
		if (centroid[axis] < splitPos) ++i; // prim goes to left side
		else { 
			std::swap(orderedGeomIndices[i], orderedGeomIndices[j--]); // prim goes to right side
		}
	}

	// stop split if one of the sides is empty
	int leftCount = i - node.start;
	if (leftCount == 0 || leftCount == node.count) return;

	// child nodes
	int leftChild = nodesUsed++;
	int rightChild = nodesUsed++;

	// setup left child
	bvhNode[leftChild].start = node.start;
	bvhNode[leftChild].count = leftCount;

	// setup right child
	bvhNode[rightChild].start = i;
	bvhNode[rightChild].count = node.count - leftCount;

	node.left = leftChild;
	node.right = rightChild;
	node.count = 0;

	UpdateNodeBounds(leftChild, geoms);
	UpdateNodeBounds(rightChild, geoms);

	// recursion
	Subdivide(leftChild, geoms);
	Subdivide(rightChild, geoms);
}