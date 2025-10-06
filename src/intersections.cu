#include "intersections.h"

#define PI 3.14159265358979323

__host__ __device__ bool bboxIntersectionTest(const Ray& ray, const glm::vec3& bmin, const glm::vec3& bmax) {
    glm::vec3 invDir = 1.0f / ray.direction;

    glm::vec3 t1 = (bmin - ray.origin) * invDir;
    glm::vec3 t2 = (bmax - ray.origin) * invDir;

    glm::vec3 tmin3 = glm::min(t1, t2);
    glm::vec3 tmax3 = glm::max(t1, t2);

    float tmin = fmaxf(tmin3.x, fmaxf(tmin3.y, tmin3.z));
    float tmax = fminf(tmax3.x, fminf(tmax3.y, tmax3.z));

    return tmax >= tmin && tmax > 0.0f && tmin < ray.t;
}

__host__ __device__ float triangleIntersectionTest(
    const Ray& ray, const Geom& tri,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& wo)
{
    glm::vec3 v0 = glm::vec3(tri.transform * glm::vec4(tri.vertices[0], 1.0f));
    glm::vec3 v1 = glm::vec3(tri.transform * glm::vec4(tri.vertices[1], 1.0f));
    glm::vec3 v2 = glm::vec3(tri.transform * glm::vec4(tri.vertices[2], 1.0f));

    glm::vec3 baryCoords;
    bool hit = glm::intersectRayTriangle(ray.origin, ray.direction, v0, v1, v2, baryCoords);

    if (!hit) {
        return -1;
    }

    float t = baryCoords.z;
    float u = baryCoords.x;
    float v = baryCoords.y;
    float w = 1.0f - u - v;

    intersectionPoint = ray.origin + ray.direction * t;

    glm::vec3 n0 = glm::normalize(glm::vec3(tri.invTranspose * glm::vec4(tri.normals[0], 0.0f)));
    glm::vec3 n1 = glm::normalize(glm::vec3(tri.invTranspose * glm::vec4(tri.normals[1], 0.0f)));
    glm::vec3 n2 = glm::normalize(glm::vec3(tri.invTranspose * glm::vec4(tri.normals[2], 0.0f)));
    normal = glm::normalize(w * n0 + u * n1 + v * n2);

    uv = w * tri.uvs[0] + u * tri.uvs[1] + v * tri.uvs[2];

    if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1) {
        printf("UV out of range: (%f, %f)\n", uv.x, uv.y);
    }


    wo = glm::dot(ray.direction, normal) > 0.0f;
    if (wo) normal = -normal;

    return t;
}

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    glm::vec2& uv,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        
        glm::vec3 objspaceIntersection = getPointOnRay(q, tmin);

        if (glm::abs(tmin_n.x) > 0.5f) {
            uv = glm::vec2(objspaceIntersection.z + 0.5f, objspaceIntersection.y + 0.5f);
        }
        else if (glm::abs(tmin_n.y) > 0.5f) {
            uv = glm::vec2(objspaceIntersection.x + 0.5f, objspaceIntersection.z + 0.5f);
        }
        else {
            uv = glm::vec2(objspaceIntersection.x + 0.5f, objspaceIntersection.y + 0.5f);
        }

        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    bool& outside)
{
    float radius = .5;
    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));
    Ray rt;
    rt.origin = ro;
    rt.direction = rd;
    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }
    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;
    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }
    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    // Calculate spherical UV coordinates
    glm::vec3 d = glm::normalize(objspaceIntersection);
    float phi = atan2(d.z, d.x);
    float theta = asin(glm::clamp(d.y, -1.0f, 1.0f));
    uv = glm::vec2((phi + PI) / (2.0f * PI), (theta + PI / 2.0f) / PI);

    return glm::length(r.origin - intersectionPoint);
}
