# CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

- Jacqueline Guan
  - [LinkedIn](https://www.linkedin.com/in/jackie-guan/)
  - [Personal website](https://jyguan18.github.io/)
- Tested on my personal laptop:
  - Windows 11 Pro 26100.4946
  - Processor AMD Ryzen 9 7945HX with Radeon Graphics
  - 32 GB RAM
  - Nvidia GeForce RTX 4080

## Introduction

This project was to implement a path tracer using CUDA. The renderer simulates realistic light transport using Monte Carlo integration and supports a variety of materials and optimizations to improve the visuals and performance.

Key implemented features include:

- Core
  - BSDF evaluation for diffuse surfaces
  - Stream compacted path termination
  - Sorting by material type
- Extended Features
  - Refraction
  - Depth of Field
  - Texture mapping and bump mapping
  - Direct lighting
  - Environment mapping
  - OBJ Import
  - Russian Roulette Path termination
  - Hierarchical Spatial Data Structures (BVH)
  - Open Image AI Denoiser

## Features

### BSDF Shading

BSDF (Bidirectional Scattering Distribution Function) models how light reflects and refracts at a surface. This project supports diffuse, specular and refractive materials. Refraction uses Snell's Law and Fresnel equations to achieve realistic glass and transparent objects. The combination of BSDFs allows for more visual diversity in rendered scenes.

| **Diffuse** | **Specular** | **Refractive** |
| :----------: | :-----------: | :-------------: |
| <img src="img/diffuse.png" width="300"/> | <img src="img/spec.png" width="300"/> | <img src="img/glass.png" width="300"/> |


### Stochastic Sampled Anitaliasing

To achieve smooth edges and avoid jagged artifacts, I implemented stochastic sampled anitaliasing. Each pixel is sampled multiple times at random sub-pixel positions, averaging the results to create softer transitions and more appealing images

| Without Anti-Aliasing      | With Stochastic Anti-Aliasing |
| -------------------------- | ----------------------------- |
| ![No AA](img/no_aa.png) | ![AA](img/aa.png)          |

### Depth of Field

Depth of field simulates a thin-lens camera model, where only objects at the focal distance are sharp and others blur progressively. This can easily create and add a cinematic quality to renders and enhance spatial depth perception.

### Texture and Bump Mapping

Textures can be sampled from image maps, allowing materials to use color and pattern information from images. In addition, bump maps perturb the surface normals based on the maps to create the illusion of detail without actually changing any of the geometry. I also implemented as simple procedural, checkerboard texture.

| Texture Mapping                 | Bump Mapping              |
| ------------------------------- | ------------------------- |
| ![Texture](img/text.png) | ![Bump](img/bmp.png) |

### Direct lighting

I also implemented direct lighitng, which explicitly samples from the light source to reduce variance and noise in well-lit areas. Because we are gathering light contributions from known sources (rather than randomly from the scene), the image converges faster compared to normal path tracing.

| Without Direct Lighting            | With Direct Lighting          |
| ---------------------------------- | ----------------------------- |
| ![No Direct](path/to/nodirect.png) | ![Direct](path/to/direct.png) |

### Environment Mapping

I added support for environment maps, which provide ambient lighting from HDR images. This allows me use immersive lighting environments rather than simple background colors when generating a scene. We get a more global illumination that feels more natural.

| Environment Map Scene              |
| ---------------------------------- |
| ![Environment Map](img/env.png) |

### Arbitrary Mesh Loading

This path tracer also supports OBJ and MTL file loading.
OBJ meshes are loaded as triangles, which are then organized into a BVH (Bouding Volume Hierarchy) for more efficient ray-scene intersection testing. This allows the rendering of more complex 3D assets with more diverse materials.

### Intel OpenImageDenoise

I implemented Intel's OpenImageDenoise for final post-processing and it produces imnages that are smoother and higher-quality with fewer samples per pixel. OopenImageDenoise intelligently removes high-frequency noise while still preserving detail and edges, which allows the output to be a lot cleaner at lower sample counts.

| Before Denoise (500 Iterations) | After Denoise (500 Iterations)  |
| ------------------------------- | ------------------------------- |
| ![Noisy](img/noise.png)      | ![Denoised](img/denoise.png) |

## Performance Analysis

### Stream Compaction Effectiveness

Stream compaction improves performance by removing terminated rays after each bounce.

### Material Sorting

By sorting rays by material type before shading, GPU divergence is reduced and memory coherence improves.

### BVH

BVH acceleration structuure dramatically reduces the intersection testing costs by culling the scene. In scenes with more complex meshes or many triangles, the improvement is a lot more noticeable.

### Russian Roulette Path Termination

With Russian Roulette path termination, it terminates rays with low-contribution, which prevents computation that wouldn't really make a difference visually.

I used four late days on this assignment.
