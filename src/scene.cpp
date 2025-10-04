#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include "../external/tiny_obj_loader.h"

#include "stb_image.h"

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromOBJ(const std::string& filepath, const glm::mat4& transform, uint32_t materialId)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str(), nullptr, true)) {
        throw std::runtime_error("Tinyobj failed to load " + filepath + ": " + warn + err);
    }
    if (!warn.empty()) {
        std::cout << "TINYOBJ WARNING: " << warn << std::endl;
    }

    glm::mat4 inverseTransform = glm::inverse(transform);
    glm::mat4 invTransposeTransform = glm::inverseTranspose(transform);

    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            Geom newTriangle;
            newTriangle.type = TRIANGLE;
            bool hasNormals = false;

            for (size_t v = 0; v < 3; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                newTriangle.vertices[v] = glm::vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                );

                if (idx.normal_index >= 0) {
                    hasNormals = true;
                    newTriangle.normals[v] = glm::vec3(
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    );
                }
            }

            if (!hasNormals) {
                glm::vec3 v0 = newTriangle.vertices[0];
                glm::vec3 v1 = newTriangle.vertices[1];
                glm::vec3 v2 = newTriangle.vertices[2];
                glm::vec3 faceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                newTriangle.normals[0] = faceNormal;
                newTriangle.normals[1] = faceNormal;
                newTriangle.normals[2] = faceNormal;
            }

            newTriangle.materialid = materialId;
            newTriangle.transform = transform;
            newTriangle.inverseTransform = inverseTransform;
            newTriangle.invTranspose = invTransposeTransform;

            this->geoms.push_back(newTriangle);

            index_offset += 3;
        }
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            const float& roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0f;
            newMaterial.hasRefractive = 0.0f;
            newMaterial.specular.color = glm::vec3(col[0], col[1],col[2]);
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Specular Transmissive") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 0.f;
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = 1.55f;
        }
        else if (p["TYPE"] == "Glass")
        {
            const auto& col = p["RGB"];
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
            newMaterial.hasRefractive = 1.0f;

        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;

        auto materialId = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];

        glm::mat4 objectTransform = utilityCore::buildTransformationMatrix(
            glm::vec3(trans[0], trans[1], trans[2]),
            glm::vec3(rotat[0], rotat[1], rotat[2]),
            glm::vec3(scale[0], scale[1], scale[2])
        );
        glm::mat4 inverseObjectTransform = glm::inverse(objectTransform);
        glm::mat4 invTransposeObjectTransform = glm::inverseTranspose(objectTransform);


        
        if (type == "mesh")
        {
            const std::string& filepath = p["FILEPATH"];

            loadFromOBJ(filepath, objectTransform, materialId);
        }
        else {
            if (type == "cube")
            {
                newGeom.type = CUBE;
            }
            else if (type == "sphere")
            {
                newGeom.type = SPHERE;
            }

            newGeom.materialid = materialId;
            newGeom.transform = objectTransform;
            newGeom.inverseTransform = inverseObjectTransform;
            newGeom.invTranspose = invTransposeObjectTransform;

            geoms.push_back(newGeom);
        }
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
