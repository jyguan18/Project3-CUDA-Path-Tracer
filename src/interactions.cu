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

        bool entering = wo.z > 0.0f;
        float etaI = entering ? etaA : etaB;
        float etaT = entering ? etaB : etaA;

        glm::vec3 n = entering ? normal : -normal;
        float eta =  etaI / etaT;

        wi = glm::refract(-wo, n, eta);
        float cosTheta = fabs(glm::dot(n, wi));
        //pathSegment.color *= m.color * (eta * eta) / glm::max(cosTheta, 1e-6f);
        if (glm::length(wi) < 1e-6f) {
            wi = glm::reflect(-wo, n);
        }
    }
    else if (m.hasRefractive > 0.5f && m.hasReflective > 0.5f) { // reflection + refraction (glass)
        float etaA = 1.0f;
        float etaB = (m.indexOfRefraction > 0.0) ? m.indexOfRefraction : 1.55f;
        bool entering = glm::dot(wo, normal) > 0.0f;
        glm::vec3 n = entering ? normal : -normal;
        float eta = entering ? (etaA / etaB) : (etaB / etaA);

        float cosTheta = glm::dot(wo, n);
        float R0 = (etaA - etaB) / (etaA + etaB);
        R0 = R0 * R0;
        float fresnel = R0 + (1.0f - R0) * powf(1.0f - cosTheta, 5.0);

        if (xi < fresnel) {
            wi = glm::reflect(-wo, n);
        }
        else {
            wi = glm::refract(-wo, n, eta);
            if (glm::length(wi) < 1e-6f) {
                wi = glm::reflect(-wo, n);
            }
        }
    }

    pathSegment.ray.origin = intersect + wi * 0.001f;
    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}
