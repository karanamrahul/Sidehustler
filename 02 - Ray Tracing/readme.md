# Simple Ray Caster Project

## Introduction

This project is a journey into the world of computer graphics, focusing on the creation of a simple ray caster from scratch. It encompasses transforming 3D scenes into captivating 2D images by simulating light interactions with objects to achieve realism in rendering.

## Image Formation: From 3D to 2D

The essence of ray casting involves projecting a 3D scene onto a 2D plane, akin to capturing a 3D environment in a 2D photograph using a virtual camera model. This process involves:

- **3D to 2D Image Plane Projection:** Mathematical transformations project each point in the 3D scene onto the 2D image plane, representing the camera's viewpoint.
- **Line of Sight:** A conceptual ray extending from the viewer's eye through the image plane into the scene, determining visible parts in the final image.
- **Projecting the Contour of 3D Objects to 2D:** Essential for defining object outlines and understanding their occlusion from the viewer's perspective.

## Color and Light: The Magic of Ray Tracing

Ray tracing simulates light's interaction with objects to produce realistic colors and shadows:

- **Forward Ray Tracing:** Traces rays from the light source to the viewer, though inefficient for image rendering.
![Forward Ray Tracing](https://github.com/karanamrahul/Sidehustler/blob/4b56ccb0dfe3d6de7dcea8bb5c8fb1f185a43f9c/02%20-%20Ray%20Tracing/tracefromeyetolight.gif)
- **Backward Ray Tracing (Ray Casting):** Starts rays at the viewer's eye, tracing outward until they hit scene objects, focusing only on rays that contribute to the visible scene.

## Key Components of Ray Casting

- **Reflection and Refraction:** Simulate realistic effects of light bouncing off or passing through surfaces.
![Reflection and Refraction](https://github.com/karanamrahul/Sidehustler/blob/4b56ccb0dfe3d6de7dcea8bb5c8fb1f185a43f9c/02%20-%20Ray%20Tracing/reflectionrefraction.gif)
- **Primary Ray:** The initial ray determining the first object intersection to calculate the pixel's basic color.
- **Shadow Ray:** Cast toward light sources after an object intersection to simulate shadows.
![Shadow Ray](https://github.com/karanamrahul/Sidehustler/blob/4b56ccb0dfe3d6de7dcea8bb5c8fb1f185a43f9c/02%20-%20Ray%20Tracing/lightingnoshadow.gif)
## Rendering: The Final Step

Rendering combines ray casting, light simulation, and shading models to produce the final image, involving:

1. **Setting up the Camera and Image Plane:** Define the camera's position, orientation, and image plane dimensions.
2. **Ray Casting:** Determine the closest object intersection for each pixel.
3. **Color Calculation:** Apply lighting, material properties, and effects like shadows and reflection to determine pixel colors.
4. **Rendering the Image:** Aggregate all pixel colors to form the final 2D image.

## Key Components

### Vector Class (`Vec3`)

A template class for 3D vectors, providing essential operations like normalization, dot product, and arithmetic operations necessary for ray tracing calculations.

### Sphere Class

Defines spheres in the scene, each with properties like position, radius, color, and material characteristics (e.g., reflection, transparency).

### Ray Tracing Logic

The core of the ray tracing logic is in the `trace` function. It works by casting rays from the eye (camera) through each pixel on the image plane and into the scene. For each ray, the function checks for intersections with objects in the scene. Based on the material properties of the intersected object and the lighting conditions, it calculates the color of the ray. This process includes handling reflections and refractions for materials with those properties.

### Rendering Function (`render`)

Iterates over the image pixels, casting rays into the scene for each pixel, and uses the `trace` function to determine the pixel colors. The final image is saved in the PPM format.

### Scene Setup

The `main` function initializes the scene with spheres and a light source. Users can modify this function to customize the scene.

### Execution

Run the compiled executable to render the scene. The output image is saved as `untitled.ppm`.

### Customizing the Scene

To create different scenes, adjust the sphere properties and their arrangement in the `main` function.

### Ray Casting Logic Explained

Ray casting is the process of shooting rays from the eye (camera) into the scene to determine what the eye sees at a particular angle. In this implementation:

- For each pixel on the image, a ray is cast from the camera, passing through the pixel and extending into the scene.
- The program checks for intersections of this ray with any of the spheres in the scene. This is done by solving the geometric problem of a ray intersecting a sphere.
- When an intersection is found, the color at the intersection point is determined based on the material properties of the sphere and the lighting conditions. This includes calculating the effects of shadows, reflections, and refractions.
- If a ray does not intersect any object, a background color is returned. In this implementation, the background is simply black.

This basic approach demonstrates the fundamental principle of ray tracing, allowing for the creation of images with realistic lighting and shading effects.

![Ray Tracing](https://github.com/karanamrahul/Sidehustler/blob/4b56ccb0dfe3d6de7dcea8bb5c8fb1f185a43f9c/02%20-%20Ray%20Tracing/raytrace1.png)
## Dependencies

- Standard C++ Compiler (e.g., GCC, Clang)

## Compilation

Compile the program using the following command:

```
c++ -o raytracer -O3 -Wall raytracer.cpp
```
This command compiles raytracer.cpp into an executable named raytracer, with optimization level 3 and all compiler warnings enabled.

### Credits

* Scratchapixel 2.0: [Ray Tracing Basics](https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-ray-tracing)
* This project is inspired by the book "Ray Tracing in One Weekend" by Peter Shirley. The code is an adaptation of the ray tracer implemented in the book, with modifications and additional features.
