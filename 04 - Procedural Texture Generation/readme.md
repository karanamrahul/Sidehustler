# Procedural Generation with 2D Noise and Patterns

Procedural generation is a method to algorithmically create data and graphics. It's widely used in computer graphics for generating textures, terrains, and other complex visuals with natural-looking variations. This tutorial delves into using 2D noise for procedural generation, explaining key concepts and demonstrating how to create compelling patterns.

## Understanding Noise Functions

Noise functions are mathematical tools that generate values with seemingly random variations but are deterministic in nature. They're fundamental in creating textures that mimic the randomness found in nature.

## Interpolation in Noise Functions

Interpolation is a technique to estimate unknown values between two known values. Its variations include:


- **Linear Interpolation**: Finds a point between two known points.
![linear](https://github.com/karanamrahul/Sidehustler/blob/4b56ccb0dfe3d6de7dcea8bb5c8fb1f185a43f9c/04%20-%20Procedural%20Texture%20Generation/images/lininterpfig.png)

- **Bilinear Interpolation**: Extends linear interpolation to two dimensions, ideal for textures.
![bilinear](https://github.com/karanamrahul/Sidehustler/blob/4b56ccb0dfe3d6de7dcea8bb5c8fb1f185a43f9c/04%20-%20Procedural%20Texture%20Generation/images/bilinearfig.png)

- **Trilinear Interpolation**: Further extends bilinear interpolation to three dimensions, useful for volumetric effects.
![trilinear](https://github.com/karanamrahul/Sidehustler/blob/4b56ccb0dfe3d6de7dcea8bb5c8fb1f185a43f9c/04%20-%20Procedural%20Texture%20Generation/images/trilinearfig.png)
## Creating 2D Noise

To generate 2D noise, we distribute random values on a grid and interpolate these values for any given point:

1. Initialize a grid with random values at each vertex.
2. Identify the grid cell containing the point of interest.
3. Interpolate the values at the cell's corners to find the noise value at the point.

### Pseudo-Code for 2D Noise Generation

```
function generate2DNoise(gridSize, seed):
    initializeRandomGrid(gridSize, seed)
    for each point in grid:
        cellCorners = findSurroundingCorners(point)
        interpolatedValue = bilinearInterpolate(cellCorners, point)
        return interpolatedValue
```

#### Bilinear Interpolation
![bilinearop](https://github.com/karanamrahul/Sidehustler/blob/4b56ccb0dfe3d6de7dcea8bb5c8fb1f185a43f9c/04%20-%20Procedural%20Texture%20Generation/images/bilinear.png)

#### Trilinear Interpolation
![trilinearop](https://github.com/karanamrahul/Sidehustler/blob/main/04%20-%20Procedural%20Texture%20Generation/images/trilinear.png)



### Smoothing Noise Transitions with Cosine and Smoothstep Functions

Achieving natural-looking patterns in noise-generated textures often requires smoothing the abrupt transitions. This can be accomplished by remapping the interpolation parameter (t) using S-curve functions like cosine and smoothstep, which provide smoother transitions between values.

### Cosine Remapping
The cosine function can be used for smoothing by adjusting the interpolation parameter t to follow an S-shaped curve, leading to more natural transitions.


**Pseudo-Code for Cosine Remapping**
```
function cosineRemap(a, b, t):
    tRemapCosine = (1 - cos(t * PI)) * 0.5
    return lerp(a, b, tRemapCosine)
```

### Smoothstep Remapping
Smoothstep is another function ideal for creating smooth gradients and softening edges in procedural textures.


**Pseudo-Code for Smoothstep Remapping**
```
function smoothstepRemap(a, b, t):
    tRemapSmoothstep = t * t * (3 - 2 * t)
    return lerp(a, b, tRemapSmoothstep)
```
![comp](https://github.com/karanamrahul/Sidehustler/blob/4b56ccb0dfe3d6de7dcea8bb5c8fb1f185a43f9c/04%20-%20Procedural%20Texture%20Generation/images/comparison.png)
Using S-curve functions like cosine and smoothstep for remapping the interpolation parameter in noise functions allows for the creation of textures with more organic and natural transitions, enhancing the realism of procedural patterns.

## Generating Patterns with 2D Noise

Using 2D noise, we can create complex patterns by manipulating noise output. Key techniques include fractal sums and turbulence.

### Fractal Sum for Terrain

The fractal sum technique layers multiple noises, each with a higher frequency and lower amplitude than the last.

#### Pseudo-Code for Fractal Sum

```
function fractalSum(point, layers, baseFrequency, baseAmplitude):
    sum = 0
    frequency = baseFrequency
    amplitude = baseAmplitude
    for layer in 1 to layers:
        sum += noise(point * frequency) * amplitude
        frequency *= 2  // Increase frequency for next layer
        amplitude /= 2  // Decrease amplitude for next layer
    return sum
```

### Turbulence for Dynamic Textures
Turbulence uses the absolute value of noise to create a bumpy effect, useful for fire, smoke, or water textures.

#### Pseudo-Code for Turbulence
```
function turbulence(point, layers, baseFrequency, baseAmplitude):
    sum = 0
    frequency = baseFrequency
    amplitude = baseAmplitude
    for layer in 1 to layers:
        sum += abs(noise(point * frequency) - 0.5) * amplitude
        frequency *= 2
        amplitude /= 2
    return sum
```


### Marble and Wood Textures

Marble textures modulate a sine wave with noise, while wood textures use multiplied noise values to simulate grain patterns.

#### Pseudo-Code for Marble Texture
```
function marbleTexture(point, noiseFrequency):
    noiseValue = noise(point * noiseFrequency)
    return (sin(point.x + noiseValue * 100) + 1) / 2
```
#### Pseudo-Code for Wood Texture

```
function woodTexture(point, noiseFrequency, grainDensity):
    noiseValue = noise(point * noiseFrequency) * grainDensity
    return noiseValue - int(noiseValue)
```
### Conclusion
This tutorial introduced procedural generation with a focus on creating patterns using 2D noise. By manipulating noise output through techniques like fractal sums and turbulence, we can simulate natural textures and patterns. Experimenting with these concepts can lead to the creation of complex and realistic graphics, opening a wide array of possibilities in computer graphics and beyond.