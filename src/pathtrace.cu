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
#define RUSSIAN_ROULETTE true

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

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
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
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    bvh.build(scene->geoms);
    printf("BVH nodes: %zu, orderedIndices: %zu, nodesUsed=%d\n", bvh.bvhNode.size(), bvh.orderedGeomIndices.size(), bvh.nodesUsed);
    for (int n = 0; n < bvh.nodesUsed; ++n) {
        auto& bn = bvh.bvhNode[n];
        printf("node %d: start=%d count=%d left=%d right=%d bboxMin=(%f,%f,%f) bboxMax=(%f,%f,%f)\n", n, bn.start, bn.count, bn.left, bn.right, bn.bboxMin.x, bn.bboxMin.y, bn.bboxMin.z, bn.bboxMax.x, bn.bboxMax.y, bn.bboxMax.z);
    }
    // After building BVH and before copying:
    for (size_t i = 0; i < bvh.orderedGeomIndices.size(); ++i) {
        int gi = bvh.orderedGeomIndices[i];
        if (gi < 0 || gi >= (int)scene->geoms.size()) {
            printf("HOST BAD ordered index %zu -> %d (geoms %zu)\n", i, gi, scene->geoms.size());
        }
    }
    for (int n = 0; n < bvh.nodesUsed; n++) {
        auto& nd = bvh.bvhNode[n];
        if (nd.count > 0) {
            assert(nd.start >= 0 && nd.start + nd.count <= bvh.orderedGeomIndices.size());
        }
        else {
            assert(nd.left >= 0 && nd.left < bvh.nodesUsed);
            assert(nd.right >= 0 && nd.right < bvh.nodesUsed);
        }
    }



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
    cudaMemcpy(dev_bvhNodes, bvh.bvhNode.data(), bvh.bvhNode.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_orderedGeomIndices, bvh.orderedGeomIndices.size() * sizeof(int));
    cudaMemcpy(dev_orderedGeomIndices, bvh.orderedGeomIndices.data(), bvh.orderedGeomIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

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

        float randX = u01(rng);
        float randY = u01(rng);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + randX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + randY - (float)cam.resolution.y * 0.5f)
        );

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

                    if (geom.type == CUBE)
                    {
                        t = boxIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
                    }
                    else if (geom.type == SPHERE)
                    {
                        t = sphereIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside);
                    }
                    else if (geom.type == TRIANGLE)
                    {
                        t = triangleIntersectionTest(r, geom, tmp_intersect, tmp_normal, outside);
                    }

                    if (t > 0.0f && t_min > t)
                    {
                        t_min = t;
                        r.t = t;
                        hit_geom_index = primitiveIdx;
                        intersect_point = tmp_intersect;
                        normal = tmp_normal;
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
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?
            else if (geom.type == TRIANGLE) {
                t = triangleIntersectionTest(pathSegment.ray, geom, tmp_intersect, tmp_normal, outside);
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

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int traceDepth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    
    if (intersection.t <= 0.0) {
        pathSegments[idx].color = glm::vec3(0.0f);
        pathSegments[idx].remainingBounces = 0;
        return; // hit nothing
    }

    if (pathSegments[idx].remainingBounces <= 0) {
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
    thrust::uniform_real_distribution<float> u01(0, 1);

    Material material = materials[intersection.materialId];
    glm::vec3 materialColor = material.color;

    // hit light
    if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
        pathSegments[idx].remainingBounces = 0;
        return;
    }

#if RUSSIAN_ROULETTE
    const int min_bounces = 3;
    int currentDepth = traceDepth - pathSegments[idx].remainingBounces;

    if (currentDepth >= min_bounces) {

        glm::vec3 throughput = pathSegments[idx].color;
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

        p = fminf(p, 0.95f);
        if (p <= 1e-6f) {
            pathSegments[idx].remainingBounces = 0;
            return;
        }

        if (u01(rng) > p) {
            pathSegments[idx].remainingBounces = 0;
            return;
        }

        pathSegments[idx].color *= 1 / p;
    }
#endif

    glm::vec3 hitPoint = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;

    glm::vec3 n = intersection.surfaceNormal;

    if (glm::dot(n, pathSegments[idx].ray.direction) > 0.0) {
        n = -n;
    }

    scatterRay(pathSegments[idx], hitPoint, n, material, rng);
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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
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

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

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
            traceDepth
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

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
