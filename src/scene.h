#pragma once

#include "sceneStructs.h"
#include <vector>
#include <unordered_map>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromOBJ(const std::string& objName,
        const glm::mat4& transform,
        uint32_t materialId);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    RenderState state;
};
