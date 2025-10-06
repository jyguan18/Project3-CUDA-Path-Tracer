#pragma once

#include "sceneStructs.h"
#include <vector>
#include <unordered_map>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromOBJ(const std::string& objName,
        const glm::mat4& transform, uint32_t materialId);
    void loadTexture(const std::string& filepath, Texture& tex, bool isFloat);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    std::vector<glm::vec3> textureData;
    RenderState state;
};
