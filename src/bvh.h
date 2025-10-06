#pragma once
#include "sceneStructs.h"
#include "intersections.h"
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>

#include "glm/glm.hpp"


#include <string>

// Axis-Aligned Bounding Box
struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    AABB() {
        min = glm::vec3(1e30f);
        max = glm::vec3(-1e30f);
    }

    void expand(const glm::vec3& point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }

    void expand(const AABB& box) {
        min = glm::min(min, box.min);
        max = glm::max(max, box.max);
    }
};

struct BVHNode {
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
    int left ;
    int right;
    int start;
    int count;
    bool isLeaf() const {
        return count > 0;
    }

    BVHNode():bboxMin(glm::vec3(1e30f)), bboxMax(glm::vec3(-1e30f)), left(-1), right(-1), count(0), start(-1) {};
};

class BVH {
public:
    void build(const std::vector<Geom>& geoms);
    void IntersectBVH(Ray& ray, const std::vector<Geom>& geoms, int nodeIdx, ShadeableIntersection& isect);
    
    std::vector<int> orderedGeomIndices;
    std::vector<BVHNode> bvhNode;

    int nodesUsed;
private:
    void UpdateNodeBounds(int idx, const std::vector<AABB>& boxes);
    void Subdivide(int idx, const std::vector<Geom>& geoms, const std::vector<AABB>& boxes);

    int rootNodeIdx;
    
};