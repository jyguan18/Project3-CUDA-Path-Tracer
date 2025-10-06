#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ float samplePointOnLight(
    const Geom& light,
    const glm::vec3& hit_point,
    glm::vec3& light_point,
    glm::vec3& light_normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (light.type == SPHERE) {
        float u1 = u01(rng);
        float u2 = u01(rng);

        float z = 1.f - 2.f * u1;
        float r = sqrtf(fmaxf(0.f, 1.f - z * z));
        float phi = 2.f * PI * u2;

        glm::vec3 pLocal(r * cosf(phi), r * sinf(phi), z); 
        glm::vec4 pWorld = light.transform * glm::vec4(pLocal, 1.f);
        light_point = glm::vec3(pWorld);

        glm::vec4 nWorld = glm::transpose(glm::inverse(light.transform)) * glm::vec4(pLocal, 0.f);
        light_normal = glm::normalize(glm::vec3(nWorld));

        float radius = glm::length(glm::vec3(light.transform[0]));
        return 4.f * PI * radius * radius;
    }
    else if (light.type == CUBE) {
        int face = int(u01(rng) * 6.f); // pick random face
        float u = u01(rng);
        float v = u01(rng);

        glm::vec3 pLocal;
        glm::vec3 nLocal;

        switch (face) {
        case 0: pLocal = glm::vec3(0.5f, u - 0.5f, v - 0.5f); nLocal = glm::vec3(1, 0, 0); break;  // +X
        case 1: pLocal = glm::vec3(-0.5f, u - 0.5f, v - 0.5f); nLocal = glm::vec3(-1, 0, 0); break; // -X
        case 2: pLocal = glm::vec3(u - 0.5f, 0.5f, v - 0.5f); nLocal = glm::vec3(0, 1, 0); break;  // +Y
        case 3: pLocal = glm::vec3(u - 0.5f, -0.5f, v - 0.5f); nLocal = glm::vec3(0, -1, 0); break; // -Y
        case 4: pLocal = glm::vec3(u - 0.5f, v - 0.5f, 0.5f); nLocal = glm::vec3(0, 0, 1); break;  // +Z
        case 5: pLocal = glm::vec3(u - 0.5f, v - 0.5f, -0.5f); nLocal = glm::vec3(0, 0, -1); break; // -Z
        }

        glm::vec4 pWorld = light.transform * glm::vec4(pLocal, 1.f);
        light_point = glm::vec3(pWorld);

        glm::vec4 nWorld = glm::transpose(glm::inverse(light.transform)) * glm::vec4(nLocal, 0.f);
        light_normal = glm::normalize(glm::vec3(nWorld));

        glm::vec3 ex = glm::vec3(light.transform * glm::vec4(1, 0, 0, 0));
        glm::vec3 ey = glm::vec3(light.transform * glm::vec4(0, 1, 0, 0));
        glm::vec3 ez = glm::vec3(light.transform * glm::vec4(0, 0, 1, 0));
        float area = 2.f * (glm::length(glm::cross(ex, ey)) +
            glm::length(glm::cross(ex, ez)) +
            glm::length(glm::cross(ey, ez)));
        return area;
    }
    else if (light.type == TRIANGLE) {
        float u = u01(rng);
        float v = u01(rng);
        if (u + v > 1.f) {
            u = 1.f - u;
            v = 1.f - v;
        }
        float w = 1.f - u - v;

        glm::vec3 v0 = glm::vec3(light.transform * glm::vec4(light.vertices[0], 1.f));
        glm::vec3 v1 = glm::vec3(light.transform * glm::vec4(light.vertices[1], 1.f));
        glm::vec3 v2 = glm::vec3(light.transform * glm::vec4(light.vertices[2], 1.f));

        light_point = w * v0 + u * v1 + v * v2;

        glm::vec3 n = glm::normalize(glm::cross(v1 - v0, v2 - v0));
        light_normal = glm::normalize(glm::vec3(light.transform * glm::vec4(n, 0.f)));

        float area = 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0));
        return area;
    }

    light_point = glm::vec3(0);
    light_normal = glm::vec3(0, 1, 0);
    return 1.f;
}



__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float xi = u01(rng);

    glm::vec3 wi;
    glm::vec3 wo = -glm::normalize(pathSegment.ray.direction);

    if (m.hasReflective < 0.5f && m.hasRefractive < 0.5f) { // diffuse
        wi = calculateRandomDirectionInHemisphere(normal, rng);
    }
    else if (m.hasReflective > 0.5f && m.hasRefractive < 0.5f) { // reflection
        wi = glm::reflect(-wo, normal);
    }
    else if (m.hasReflective < 0.5f && m.hasRefractive > 0.5f) { // Specular transmission
        float etaA = 1.;
        float etaB = 1.55;

        bool entering = glm::dot(wo, normal) > 0.0f;
        float etaI = entering ? etaA : etaB;
        float etaT = entering ? etaB : etaA;

        glm::vec3 n = entering ? normal : -normal;
        float eta =  etaI / etaT;

        wi = glm::refract(-wo, n, eta);

        if (glm::length(wi) < 1e-6f) {
            wi = glm::reflect(-wo, n);
        }
    }
    else if (m.hasRefractive > 0.5f && m.hasReflective > 0.5f) { // glass
        float etaA = 1.0f;
        float etaB = (m.indexOfRefraction > 0.0) ? m.indexOfRefraction : 1.55f;

        bool entering = glm::dot(pathSegment.ray.direction, normal) < 0.0f;
        glm::vec3 n = entering ? normal : -normal;
        float eta = entering ? (etaA / etaB) : (etaB / etaA);

        float cosTheta = glm::abs(glm::dot(pathSegment.ray.direction, n));
        float R0 = (etaA - etaB) / (etaA + etaB);
        R0 = R0 * R0;
        float fresnel = R0 + (1.0f - R0) * powf(1.0f - cosTheta, 5.0f);

        if (xi < fresnel) {
            wi = glm::reflect(pathSegment.ray.direction, n);  // Don't negate!
        }
        else {
            wi = glm::refract(pathSegment.ray.direction, n, eta);  // Don't negate!
            if (glm::length(wi) < 1e-6f) {
                wi = glm::reflect(pathSegment.ray.direction, n);
            }
        }
    }

    pathSegment.ray.origin = intersect + wi * 0.001f;
    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}
