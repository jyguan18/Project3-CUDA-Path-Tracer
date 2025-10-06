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
			glm::vec2 tmp_uv;
			if (geoms[geomIdx].type == TRIANGLE) {
				t = triangleIntersectionTest(ray, geoms[geomIdx], intersectionPoint, normal, tmp_uv, wo);
			}
			else if (geoms[geomIdx].type == SPHERE) {
				t = sphereIntersectionTest(geoms[geomIdx], ray, intersectionPoint, normal, tmp_uv, wo);
			}
			else if (geoms[geomIdx].type == CUBE) {
				t = boxIntersectionTest(geoms[geomIdx], ray, intersectionPoint, normal, tmp_uv, wo);
			}
			if (t > 0 && t < ray.t) {
				ray.t = t;
				isect.t = t;
				isect.surfaceNormal = normal;
				isect.materialId = geoms[geomIdx].materialid;
				isect.uv = tmp_uv;
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

	std::vector<AABB> precomputedBoxes(geoms.size());
	for (int i = 0; i < geoms.size(); ++i) {
		precomputedBoxes[i] = boundingBox(geoms[i]);
	}

	UpdateNodeBounds(rootNodeIdx, precomputedBoxes); // makes the bounding box of the root node
	Subdivide(rootNodeIdx, geoms, precomputedBoxes); // recursively subdivides the root node to build out the tree
}

void BVH::UpdateNodeBounds(int idx, const std::vector<AABB>& boxes) {
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
		node.count = std::max(0, Nidx - node.start);
	}

	for (int i = 0; i < node.count; i++) {
		int geomIdx = orderedGeomIndices[node.start + i];
		const AABB& box = boxes[geomIdx];  // Use precomputed box
		node.bboxMin = glm::min(node.bboxMin, box.min);
		node.bboxMax = glm::max(node.bboxMax, box.max);
	}
}

void BVH::Subdivide(int idx, const std::vector<Geom>& geoms, const std::vector<AABB>& boxes) {
	BVHNode& node = bvhNode[idx];

	if (node.count <= 2) return;

	glm::vec3 extent = node.bboxMax - node.bboxMin;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;

	int mid = node.start + (node.count / 2);
	std::nth_element(
		orderedGeomIndices.begin() + node.start, 
		orderedGeomIndices.begin() + mid,         
		orderedGeomIndices.begin() + node.start + node.count,
		[&](int a_idx, int b_idx) {
			const AABB& boxA = boxes[a_idx];
			const AABB& boxB = boxes[b_idx];

			glm::vec3 centroidA = (boxA.min + boxA.max) * 0.5f;
			glm::vec3 centroidB = (boxB.min + boxB.max) * 0.5f;

			return centroidA[axis] < centroidB[axis];
		});

	int leftCount = mid - node.start;
	int rightCount = node.count - leftCount;

	if (leftCount == 0 || rightCount == 0) return;


	int leftChildIdx = nodesUsed++;
	int rightChildIdx = nodesUsed++;

	bvhNode[leftChildIdx].start = node.start;
	bvhNode[leftChildIdx].count = leftCount;

	bvhNode[rightChildIdx].start = mid;
	bvhNode[rightChildIdx].count = rightCount;

	node.left = leftChildIdx;
	node.right = rightChildIdx;
	node.count = 0;

	UpdateNodeBounds(leftChildIdx, boxes);
	UpdateNodeBounds(rightChildIdx, boxes);

	Subdivide(leftChildIdx, geoms, boxes);
	Subdivide(rightChildIdx, geoms, boxes);
}