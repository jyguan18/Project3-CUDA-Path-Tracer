#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "bvh.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define SORT_MAT true
#define COMPACT true
#define ENABLE_BVH true
#define RUSSIAN_ROULETTE false
#define DENOISE false
#define DENOISE_INTERVAL 5

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        // avoid divide-by-zero
        float denom = (iter > 0) ? float(iter) : 1.0f;

        int rx = glm::clamp((int)(pix.x / denom * 255.0f), 0, 255);
        int gx = glm::clamp((int)(pix.y / denom * 255.0f), 0, 255);
        int bx = glm::clamp((int)(pix.z / denom * 255.0f), 0, 255);

        pbo[index].w = 0;
        pbo[index].x = rx;
        pbo[index].y = gx;
        pbo[index].z = bx;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static BVH bvh;
static BVHNode* dev_bvhNodes = NULL;
static int* dev_orderedGeomIndices = NULL;
static int* dev_light_indices = NULL;

static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;
static glm::vec3* dev_oidn_image = NULL;

static int hst_light_count = 0;
static oidn::DeviceRef oidn_device;
static oidn::FilterRef oidn_color_filter;
static oidn::FilterRef oidn_albedo_filter;
static oidn::FilterRef oidn_normal_filter;

static glm::vec3* dev_textures = NULL;
void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void denoiserInit() {
    // Create OIDN device (CUDA) and use the global oidn_device variable
    oidn_device = oidn::newDevice(oidn::DeviceType::CUDA);

    // Choose CUDA device 0 explicitly (change if you need device 1 etc.)
    oidn_device.set("device", 0);

    // Commit the device (required before creating filters)
    oidn_device.commit();

    // Optional: check device errors early
    const char* devErr = nullptr;
    if (oidn_device.getError(devErr) != oidn::Error::None) {
        std::cerr << "OIDN device error after commit: " << (devErr ? devErr : "(null)") << std::endl;
        return;
    }

    const int width = hst_scene->state.camera.resolution.x;
    const int height = hst_scene->state.camera.resolution.y;

    // Make sure your dev_albedo/dev_normal/dev_oidn_image are already cudaMalloc'ed
    // (you allocate them in pathtraceInit before calling denoiserInit())

    // --- Color Filter (Main) ---
    oidn_color_filter = oidn_device.newFilter("RT");
    oidn_color_filter.setImage("color", dev_image, oidn::Format::Float3, width, height);
    oidn_color_filter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
    oidn_color_filter.setImage("normal", dev_normal, oidn::Format::Float3, width, height);
    oidn_color_filter.setImage("output", dev_oidn_image, oidn::Format::Float3, width, height);
    oidn_color_filter.set("hdr", true);       // HDR colors
    oidn_color_filter.set("cleanAux", true);  // denoise guides first
    oidn_color_filter.commit();

    // --- Albedo Filter ---
    oidn_albedo_filter = oidn_device.newFilter("RT");
    oidn_albedo_filter.setImage("color", dev_albedo, oidn::Format::Float3, width, height); // filter input = albedo
    oidn_albedo_filter.setImage("output", dev_albedo, oidn::Format::Float3, width, height); // write back to dev_albedo
    oidn_albedo_filter.commit();

    // --- Normal Filter ---
    oidn_normal_filter = oidn_device.newFilter("RT");
    oidn_normal_filter.setImage("color", dev_normal, oidn::Format::Float3, width, height); // filter input = normal
    oidn_normal_filter.setImage("output", dev_normal, oidn::Format::Float3, width, height); // write back to dev_normal
    oidn_normal_filter.commit();

    // Check for errors after creating/committing filters
    if (oidn_device.getError(devErr) != oidn::Error::None) {
        std::cerr << "OIDN device error after filter creation: " << (devErr ? devErr : "(null)") << std::endl;
    }
}



void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    std::vector<int> hst_light_indices;
    for (int i = 0; i < scene->geoms.size(); ++i) {
        const Geom& g = scene->geoms[i];
        const Material& m = scene->materials[g.materialid];
        if (m.emittance > 0.0f) {
            hst_light_indices.push_back(i);
        }
    }
    hst_light_count = hst_light_indices.size();
    if (hst_light_count > 0) {
        cudaMalloc(&dev_light_indices, hst_light_count * sizeof(int));
        cudaMemcpy(dev_light_indices, hst_light_indices.data(), hst_light_count * sizeof(int), cudaMemcpyHostToDevice);
    }

    bvh.build(scene->geoms);

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_bvhNodes, bvh.bvhNode.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, bvh.bvhNode.data(), bvh.nodesUsed * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_orderedGeomIndices, bvh.orderedGeomIndices.size() * sizeof(int));
    cudaMemcpy(dev_orderedGeomIndices, bvh.orderedGeomIndices.data(), bvh.orderedGeomIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));

#if DENOISE
    
    cudaMalloc(&dev_oidn_image, pixelcount * sizeof(glm::vec3));

    denoiserInit();
#endif

    if (scene->textureData.size() > 0) {  // Use textureData, not textures
        cudaMalloc(&dev_textures, scene->textureData.size() * sizeof(glm::vec3));
        cudaMemcpy(dev_textures, scene->textureData.data(),
            scene->textureData.size() * sizeof(glm::vec3),
            cudaMemcpyHostToDevice);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    cudaFree(dev_bvhNodes);
    cudaFree(dev_orderedGeomIndices);
    cudaFree(dev_albedo);
    cudaFree(dev_normal);

#if DENOISE
    
    cudaFree(dev_oidn_image);
    // OIDN objects release themselves automatically
#endif

    cudaFree(dev_textures);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    //RNG rng = makeSeededRandomEngine(iter, index, 0);

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
        thrust::uniform_real_distribution<float> u02(0, 1);

        float randX = u01(rng);
        float randY = u01(rng);

        // TODO: implement antialiasing by jittering the ray
        glm::vec3 pixelDir = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + randX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + randY - (float)cam.resolution.y * 0.5f)
        );

        // DOF start... getting the focal point
        glm::vec3 focalPoint = cam.position + pixelDir * cam.focalDistance;

        float angle = u02(rng) * 2.0f * PI;
        float r = glm::sqrt(u02(rng)) * cam.apertureRadius;
        glm::vec2 diskSample = glm::vec2(r * glm::cos(angle), r * glm::sin(angle));

        glm::vec3 offset = cam.right * diskSample.x + cam.up * diskSample.y;

        segment.ray.origin = cam.position + offset;
        segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}


__device__ void traverseBVH(
    int path_index,
    Ray& r,
    BVHNode* nodes,
    int* indices,
    Geom* geoms,
    ShadeableIntersection* intersections)
{
    int stack[64] = { 0 };
    int idx = 0;

    int nodeIdx = 0;

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    while (nodeIdx >= 0)
    {
        BVHNode& node = nodes[nodeIdx];
        if (bboxIntersectionTest(r, node.bboxMin, node.bboxMax )) {
            if (node.count > 0) {
                for (int i = 0; i < node.count; ++i)
                {
                    int primitiveIdx = indices[node.start + i];

                    Geom& geom = geoms[primitiveIdx];
                    glm::vec2 tmp_uv;

                    if (geom.type == CUBE)
                    {
                        t = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_uv, outside);
                    }
                    else if (geom.type == SPHERE)
                    {
                        t = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_uv, outside);
                    }
                    else if (geom.type == TRIANGLE)
                    {
                        t = triangleIntersectionTest(r, geom, tmp_intersect, tmp_normal, tmp_uv, outside);
                    }

                    if (t > 0.0f && t_min > t)
                    {
                        t_min = t;
                        r.t = t;
                        hit_geom_index = primitiveIdx;
                        intersect_point = tmp_intersect;
                        normal = tmp_normal;
                        intersections[path_index].uv = tmp_uv;
                    }
                }

                if (idx > 0) {
                    nodeIdx = stack[--idx];  // remove from stack
                }
                else {

                    nodeIdx = -1;
                    break; // end if stack is empty
                }
            }
            else {
                stack[idx++] = node.right;
                nodeIdx = node.left;
            }
        }
        else {
            if (idx > 0) {
                nodeIdx = stack[--idx];
            }
            else {
                nodeIdx = -1;
                break;
            }
        }
    }

    if (hit_geom_index == -1)
    {
        intersections[path_index].t = -1.0f;
    }
    else
    {
        intersections[path_index].t = t_min;
        intersections[path_index].materialId = geoms[hit_geom_index].materialid;
        intersections[path_index].surfaceNormal = normal;
    }

    
}

__device__ float traceShadowRay(Ray& shadow_ray, BVHNode* nodes, int* indices, Geom* geoms)
{
    ShadeableIntersection shadow_isect;
    float saved_t = shadow_ray.t;

    traverseBVH(0, shadow_ray, nodes, indices, geoms, &shadow_isect);
    if (shadow_isect.t > 0.0f && shadow_isect.t < saved_t - 1e-4f) return shadow_isect.t;

    return -1.0f;
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    BVHNode* nodes,
    int* indices,
    int numNodes,
    int numIndices)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];
        pathSegment.ray.t = FLT_MAX;

#if ENABLE_BVH
        traverseBVH(path_index, pathSegment.ray, nodes, indices, geoms, intersections);
#else
        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?
            else if (geom.type == TRIANGLE) {
                t = triangleIntersectionTest(pathSegment.ray, geom, tmp_intersect, tmp_normal, tmp_uv, outside);
            }
            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && pathSegment.ray.t > t)
            {
                pathSegment.ray.t = t;
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = pathSegment.ray.t;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
#endif
    }
}

__device__ void addDirectLighting(
    int pixel_index,
    const glm::vec3& hitPoint,
    const glm::vec3& normal,
    const Material& material,
    const glm::vec3& throughput,
    Geom* geoms,
    Material* materials,
    BVHNode* nodes,
    int* bvh_indices,
    int* light_indices,
    int light_count,
    glm::vec3* image,
    thrust::default_random_engine& rng)
{
    if (light_count == 0) return;

    thrust::uniform_int_distribution<int> u_light(0, light_count - 1);
    int light_geom_idx = light_indices[u_light(rng)];
    const Geom& light = geoms[light_geom_idx];
    const Material& light_mat = materials[light.materialid];

    glm::vec3 point_on_light, light_normal;
    float light_area = samplePointOnLight(light, hitPoint, point_on_light, light_normal, rng);

    Ray shadow_ray;
    shadow_ray.origin = hitPoint + normal * 0.001f;
    glm::vec3 dir_to_light = point_on_light - shadow_ray.origin;
    float dist = glm::length(dir_to_light);
    shadow_ray.direction = dir_to_light / dist;
    shadow_ray.t = dist - 0.001f;

    float shadow_t = traceShadowRay(shadow_ray, nodes, bvh_indices, geoms);
    if (shadow_t >= 0.0f) {
        return;
    }

    float cos_theta_surface = glm::max(0.0f, glm::dot(normal, shadow_ray.direction));
    float cos_theta_light = glm::max(0.0f, glm::dot(light_normal, -shadow_ray.direction));

    if (cos_theta_surface <= 0.0f || cos_theta_light <= 0.0f) {
        return;
    }

    glm::vec3 bsdf = material.color / PI;
    float G = cos_theta_light / (dist * dist + 1e-8f);

    float pdf_area = 1.0f / (light_area + 1e-8f);

    glm::vec3 Le = light_mat.color * light_mat.emittance;

    glm::vec3 direct = throughput * Le * bsdf * cos_theta_surface * G / pdf_area * (float)light_count;
    float max_contribution = 10.0f;
    direct = glm::min(direct, glm::vec3(max_contribution));

    atomicAdd(&image[pixel_index].x, direct.x);
    atomicAdd(&image[pixel_index].y, direct.y);
    atomicAdd(&image[pixel_index].z, direct.z);
}

__device__ glm::vec3 sampleTexture(const Texture& tex, const glm::vec2& uv, glm::vec3* textures) {
    if (tex.index < 0) return glm::vec3(1.0f);

    int x = glm::clamp(int(uv.x * (tex.width - 1)), 0, tex.width - 1);
    int y = glm::clamp(int(uv.y * (tex.height - 1)), 0, tex.height - 1);

    if (x < 0) x += tex.width;
    if (y < 0) y += tex.height;

    int idx = tex.startIdx + y * tex.width + x;
    return textures[idx];
}

__device__ glm::vec3 sampleEnvironmentMap(
    const glm::vec3& direction,
    const Texture& envMap,
    glm::vec3* textures)
{
    if (envMap.index < 0) return glm::vec3(0.0f);

    float phi = atan2(direction.z, direction.x);
    float theta = acos(glm::clamp(direction.y, -1.0f, 1.0f));

    glm::vec2 uv;
    uv.x = (phi + PI) / (2.0f * PI);
    uv.y = theta / PI;

    return sampleTexture(envMap, uv, textures);
}

__device__ glm::vec3 proceduralCheckerboard(const glm::vec2& uv, float scale) {
    float u = uv.x * scale;
    float v = uv.y * scale;

    bool checkerU = (int(floorf(u)) & 1) == 0;
    bool checkerV = (int(floorf(v)) & 1) == 0;

    return (checkerU ^ checkerV) ? glm::vec3(0.9f) : glm::vec3(0.1f);
}

__device__ glm::vec3 applyBumpMapping(
    const glm::vec2& uv,
    const glm::vec3& normal,
    const Texture& bumpTex,
    float bumpStrength,
    glm::vec3* textures)
{
    if (bumpTex.index < 0) return normal;

    float du = 1.0f / bumpTex.width;
    float dv = 1.0f / bumpTex.height;

    glm::vec3 hC = sampleTexture(bumpTex, uv, textures);
    glm::vec3 hR = sampleTexture(bumpTex, uv + glm::vec2(du, 0), textures);
    glm::vec3 hU = sampleTexture(bumpTex, uv + glm::vec2(0, dv), textures);

    float heightC = (hC.x + hC.y + hC.z) / 3.0f;
    float heightR = (hR.x + hR.y + hR.z) / 3.0f;
    float heightU = (hU.x + hU.y + hU.z) / 3.0f;

    float dh_du = (heightR - heightC) * bumpStrength;
    float dh_dv = (heightU - heightC) * bumpStrength;

    glm::vec3 tangent = glm::normalize(glm::cross(normal, glm::vec3(0, 1, 0)));
    if (glm::length(tangent) < 0.01f) {
        tangent = glm::normalize(glm::cross(normal, glm::vec3(1, 0, 0)));
    }
    glm::vec3 bitangent = glm::cross(normal, tangent);

    glm::vec3 perturbedNormal = normal - dh_du * tangent - dh_dv * bitangent;
    return glm::normalize(perturbedNormal);
}

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int traceDepth,
    glm::vec3* image,
    Geom* geoms,
    BVHNode* nodes,
    int* bvh_indices,
    int* light_indices,
    int light_count,
    glm::vec3* dev_albedo,
    glm::vec3* dev_normal,
    glm::vec3* dev_textures,
    Texture envMap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t <= 0.0f) {
        // add environment lighting when ray does not hit anything
        glm::vec3 envColor = sampleEnvironmentMap(
            pathSegments[idx].ray.direction,
            envMap,
            dev_textures);
        
        atomicAdd(&image[pathSegments[idx].pixelIndex].x, (pathSegments[idx].color * envColor).x);
        atomicAdd(&image[pathSegments[idx].pixelIndex].y, (pathSegments[idx].color * envColor).y);
        atomicAdd(&image[pathSegments[idx].pixelIndex].z, (pathSegments[idx].color * envColor).z);
        pathSegments[idx].remainingBounces = 0;
        return;
    }

    if (pathSegments[idx].remainingBounces <= 0) {
        pathSegments[idx].remainingBounces = 0;
        return;
    }

    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 throughput = pathSegments[idx].color;
    glm::vec3 hitPoint = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;
    glm::vec3 normal = intersection.surfaceNormal;
    Material material = materials[intersection.materialId];
    glm::vec3 baseColor = material.color;


    if (material.diffuseTexture.index >= 0) {
        baseColor = sampleTexture(material.diffuseTexture, intersection.uv, dev_textures);
    }
    else if (material.useProceduralTexture) {
        baseColor *= proceduralCheckerboard(intersection.uv, 8.0f);
    }

    glm::vec3 perturbedNormal = applyBumpMapping(
        intersection.uv, normal, material.bumpTexture,
        material.bumpStrength, dev_textures);

    if (material.emittance > 0.0f) {
        glm::vec3 emission = throughput * (baseColor * material.emittance);
        atomicAdd(&image[pathSegments[idx].pixelIndex].x, emission.x);
        atomicAdd(&image[pathSegments[idx].pixelIndex].y, emission.y);
        atomicAdd(&image[pathSegments[idx].pixelIndex].z, emission.z);
        pathSegments[idx].remainingBounces = 0;
        return;
    }

    bool isSpecular = (material.hasReflective > 0.5f) || (material.hasRefractive > 0.5f);
    if (!isSpecular) {
        Material texturedMaterial = material;
        texturedMaterial.color = baseColor;

        addDirectLighting(
            pathSegments[idx].pixelIndex,
            hitPoint, perturbedNormal, texturedMaterial, throughput,
            geoms, materials, nodes, bvh_indices,
            light_indices, light_count, image, rng);
    }

    // Russian Roulette
#if RUSSIAN_ROULETTE
    const int min_bounces = 3;
    int currentDepth = traceDepth - pathSegments[idx].remainingBounces;

    if (currentDepth >= min_bounces) {
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        p = fminf(p, 0.95f);

        if (p <= 1e-6f || u01(rng) > p) {
            pathSegments[idx].remainingBounces = 0;
            return;
        }

        pathSegments[idx].color *= 1.0f / p;
    }
#endif
    material.color = baseColor;
    scatterRay(pathSegments[idx], hitPoint, perturbedNormal, material, rng);

    dev_albedo[pathSegments[idx].pixelIndex] = pathSegments[idx].color;
    dev_normal[pathSegments[idx].pixelIndex] = intersection.surfaceNormal;
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

struct IsAlive {
    __host__ __device__
        bool operator()(const PathSegment& ps) const {
        return ps.remainingBounces > 0;
    }
};

struct CompareMatId {
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};

__global__ void debug_fillBuffer(glm::vec3* buffer, glm::ivec2 resolution) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        buffer[index] = glm::vec3(1.0f, 0.0f, 1.0f); // Bright Pink!
    }
}

void OIDN_Denoise() {
    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    const int width = hst_scene->state.camera.resolution.x;
    const int height = hst_scene->state.camera.resolution.y;

    oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color", dev_image, oidn::Format::Float3, width, height);
    filter.setImage("normal", dev_normal, oidn::Format::Float3, width, height);
    filter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
    filter.setImage("output", dev_oidn_image, oidn::Format::Float3, width, height);
    filter.set("hdr", true);
    filter.set("cleanAux", true);
    filter.commit();

    oidn::FilterRef albedoFilter = device.newFilter("RT");
    albedoFilter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
    albedoFilter.setImage("output", dev_albedo, oidn::Format::Float3, width, height);
    albedoFilter.commit();

    oidn::FilterRef normalFilter = device.newFilter("RT");
    normalFilter.setImage("normal", dev_normal, oidn::Format::Float3, width, height);
    normalFilter.setImage("output", dev_normal, oidn::Format::Float3, width, height);
    normalFilter.commit();

    albedoFilter.execute();
    normalFilter.execute();
    filter.execute();

    const char* errorMessage;
    if (oidn_device.getError(errorMessage) != oidn::Error::None) {
        std::cerr << "OIDN Error: " << errorMessage << std::endl;
    }
}
__global__ void debug_fillBuffer_solid(glm::vec3* buffer, glm::ivec2 resolution, glm::vec3 color) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        buffer[index] = color;
    }
}

void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    bool iterationComplete = false;

    while (depth < traceDepth && num_paths > 0)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        int numNodes = (int)bvh.bvhNode.size();
        int numIndices = (int)bvh.orderedGeomIndices.size();

        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_bvhNodes,
            dev_orderedGeomIndices,
            numNodes,
            numIndices
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

#if SORT_MAT
        thrust::sort_by_key(
            thrust::device,
            dev_intersections,
            dev_intersections + num_paths,
            dev_paths,
            CompareMatId());
#endif

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            traceDepth,
            dev_image,
            dev_geoms,
            dev_bvhNodes,
            dev_orderedGeomIndices,
            dev_light_indices,
            hst_light_count,
            dev_albedo,
            dev_normal,
            dev_textures,
            hst_scene->environmentMap
            );

#if COMPACT
        dev_path_end = thrust::stable_partition(
            thrust::device,
            dev_paths,
            dev_path_end,
            IsAlive());
        num_paths = dev_path_end - dev_paths;
#endif

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

        depth++;
    }
    
#if DENOISE
    glm::vec3* image_to_display = dev_image;
    int display_iter = iter;

    if (iter >= 1 && ((iter % DENOISE_INTERVAL == 0) || (iter >= hst_scene->state.iterations))) {
        OIDN_Denoise();
        image_to_display = dev_oidn_image;
        dev_image = image_to_display;
        display_iter = iter;
    }

    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, display_iter, image_to_display);
#else
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
#endif

    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
