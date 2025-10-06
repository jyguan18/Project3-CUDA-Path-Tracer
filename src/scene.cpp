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

void Scene::loadFromOBJ(const std::string& filepath, const glm::mat4& transform, uint32_t jsonMaterialId)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::string mtl_basedir = filepath.substr(0, filepath.find_last_of('/') + 1);

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str(), mtl_basedir.c_str(), true)) {
        throw std::runtime_error("Tinyobj failed to load " + filepath + ": " + warn + err);
    }

    if (!warn.empty()) {
        std::cout << "TINYOBJ WARNING: " << warn << std::endl;
    }

    std::unordered_map<std::string, uint32_t> mtlNameToSceneID;
    int materialIndexOffset = this->materials.size();
    int texIdCounter = 0;

    for (const auto& mtl : materials) {
        Material newMaterial{};
        newMaterial.color = glm::vec3(mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2]);
        newMaterial.specular.color = glm::vec3(mtl.specular[0], mtl.specular[1], mtl.specular[2]);

        if (!mtl.diffuse_texname.empty() && mtl.diffuse_texname.find("_nor") == std::string::npos) {
            std::string texPath = mtl_basedir + mtl.diffuse_texname;
            loadTexture(texPath, newMaterial.diffuseTexture, false);
            newMaterial.diffuseTexture.index = texIdCounter++;
        }

        if (!mtl.bump_texname.empty()) {
            std::string bumpPath = mtl_basedir + mtl.bump_texname;
            loadTexture(bumpPath, newMaterial.bumpTexture, true);
            newMaterial.bumpTexture.index = texIdCounter++;
        }

        //if (glm::length(newMaterial.specular.color) > 1e-6f) {
        //    if (mtl.shininess > 1e-6f) {
        //        newMaterial.microfacet.isMicrofacet = true;
        //        newMaterial.microfacet.roughness = glm::min(0.8f, 1.f / glm::sqrt(mtl.shininess + 1.f));
        //    }
        //    else {t
        //        newMaterial.hasReflective = 1.f;
        //    }
        //}

        uint32_t newID = this->materials.size();
        this->materials.push_back(newMaterial);
        mtlNameToSceneID[mtl.name] = newID;
    }

    glm::mat4 invTransform = glm::inverse(transform);
    glm::mat4 invTransposeTransform = glm::inverseTranspose(transform);

    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            Geom tri{};
            tri.type = TRIANGLE;
            bool hasNormals = false;

            int mtl_index = shape.mesh.material_ids[f];
            tri.materialid = (mtl_index >= 0) 
                ? (materialIndexOffset + mtl_index)
                : jsonMaterialId;

            for (size_t v = 0; v < 3; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                tri.vertices[v] = glm::vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                );

                if (idx.normal_index >= 0) {
                    hasNormals = true;
                    tri.normals[v] = glm::vec3(
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    );
                }

                if (idx.texcoord_index >= 0) {
                    float tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                    float ty = attrib.texcoords[2 * idx.texcoord_index + 1];
                    tri.uvs[v] = glm::vec2(tx, ty);
                }
                else {
                    tri.uvs[v] = glm::vec2(0.0f);
                }
            }

            if (!hasNormals) {
                glm::vec3 faceNormal = glm::normalize(glm::cross(tri.vertices[1] - tri.vertices[0],
                    tri.vertices[2] - tri.vertices[0]));
                tri.normals[0] = tri.normals[1] = tri.normals[2] = faceNormal;
            }

            tri.transform = transform;
            tri.inverseTransform = invTransform;
            tri.invTranspose = invTransposeTransform;

            geoms.push_back(tri);
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

        if (p.contains("DIFFUSE_TEXTURE")) {
            std::string texPath = p["DIFFUSE_TEXTURE"];
            loadTexture(texPath, newMaterial.diffuseTexture, false);
        }
        if (p.contains("BUMP_TEXTURE")) {
            std::string texPath = p["BUMP_TEXTURE"];
            loadTexture(texPath, newMaterial.bumpTexture, true);
            if (p.contains("BUMP_STRENGTH")) {
                newMaterial.bumpStrength = p["BUMP_STRENGTH"];
            }
        }
        if (p.contains("USE_PROCEDURAL")) {
            newMaterial.useProceduralTexture = p["USE_PROCEDURAL"];
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];

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
            continue;
        }
        else {
            Geom newGeom;

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

void Scene::loadTexture(const std::string& filepath, Texture& tex, bool isFloat) {
    std::string fixedPath = filepath;
    std::replace(fixedPath.begin(), fixedPath.end(), '\\', '/');

    int width, height, channels;

    if (isFloat) {
        float* data = stbi_loadf(fixedPath.c_str(), &width, &height, &channels, 3);
        if (!data) {
            std::cerr << "Failed to load float texture: " << fixedPath << "\n";
            tex.index = -1;
            return;
        }
        tex.width = width;
        tex.height = height;
        tex.startIdx = textureData.size();
        tex.index = 0;
        for (int i = 0; i < width * height; i++) {
            glm::vec3 c(data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]);
            textureData.push_back(c);
        }
        stbi_image_free(data);
    }
    else {
        unsigned char* data = stbi_load(fixedPath.c_str(), &width, &height, &channels, 3);
        if (!data) {
            std::cerr << "Failed to load 8-bit texture: " << fixedPath << "\n";
            tex.index = -1;
            return;
        }
        tex.width = width;
        tex.height = height;
        tex.startIdx = textureData.size();
        tex.index = 0;
        for (int i = 0; i < width * height; i++) {
            glm::vec3 c(
                data[i * 3 + 0] / 255.0f,
                data[i * 3 + 1] / 255.0f,
                data[i * 3 + 2] / 255.0f
            );
            textureData.push_back(c);
        }
        stbi_image_free(data);
    }

    std::cout << "Loaded texture: " << fixedPath
        << " (w=" << tex.width << ", h=" << tex.height
        << ") startIdx=" << tex.startIdx
        << " size now=" << textureData.size() << "\n";
}
