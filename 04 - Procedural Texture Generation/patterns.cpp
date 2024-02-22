#include <cmath>
#include <cstdio>
#include <random>
#include <functional>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

template<typename T>
class Vec2 {
public:
    Vec2() : x(T(0)), y(T(0)) {}
    Vec2(T xx, T yy) : x(xx), y(yy) {}
    Vec2 operator * (const T &r) const { return Vec2(x * r, y * r); }
    T x, y;
};

typedef Vec2<float> Vec2f;

template<typename T = float>
inline T lerp(const T &lo, const T &hi, const T &t) {
    return lo * (1 - t) + hi * t;
}

inline float smoothstep(const float &t) {
    return t * t * (3 - 2 * t);
}

class ValueNoise {
public:
    ValueNoise(unsigned seed = 2016) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> distrFloat;
        auto randFloat = std::bind(distrFloat, gen);

        for (unsigned k = 0; k < kMaxTableSize; ++k) {
            r[k] = randFloat();
            permutationTable[k] = k;
        }

        std::uniform_int_distribution<unsigned> distrUInt;
        auto randUInt = std::bind(distrUInt, gen);
        for (unsigned k = 0; k < kMaxTableSize; ++k) {
            unsigned i = randUInt() % kMaxTableSize;
            std::swap(permutationTable[k], permutationTable[i]);
            permutationTable[k + kMaxTableSize] = permutationTable[k];
        }
    }

    float eval(const Vec2f &p) const {
        int xi = std::floor(p.x), yi = std::floor(p.y);
        float tx = p.x - xi, ty = p.y - yi;
        int rx0 = xi & kMaxTableSizeMask, ry0 = yi & kMaxTableSizeMask;
        int rx1 = (rx0 + 1) & kMaxTableSizeMask, ry1 = (ry0 + 1) & kMaxTableSizeMask;

        const float &c00 = r[permutationTable[permutationTable[rx0] + ry0]];
        const float &c10 = r[permutationTable[permutationTable[rx1] + ry0]];
        const float &c01 = r[permutationTable[permutationTable[rx0] + ry1]];
        const float &c11 = r[permutationTable[permutationTable[rx1] + ry1]];

        float sx = smoothstep(tx), sy = smoothstep(ty);
        float nx0 = lerp(c00, c10, sx), nx1 = lerp(c01, c11, sx);
        return lerp(nx0, nx1, sy);
    }

    static const unsigned kMaxTableSize = 256;
    static const unsigned kMaxTableSizeMask = kMaxTableSize - 1;
    float r[kMaxTableSize];
    unsigned permutationTable[kMaxTableSize * 2];
};
// Helper function to save a pattern to a PPM file
void savePPM(const std::string &filename, const std::vector<float> &image, int width, int height) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        unsigned char val = static_cast<unsigned char>(255.0f * std::clamp(image[i], 0.0f, 1.0f));
        ofs << val << val << val; // RGB
    }
}

// Function prototypes for generating different noise patterns
std::vector<float> generateValueNoise(const ValueNoise& noise, int width, int height, float frequency);
std::vector<float> generateFractalNoise(const ValueNoise& noise, int width, int height, float baseFrequency, int numLayers, float frequencyMult, float amplitudeMult);
std::vector<float> generateTurbulenceNoise(const ValueNoise& noise, int width, int height, float baseFrequency, int numLayers, float frequencyMult, float amplitudeMult);
std::vector<float> generateMarbleNoise(const ValueNoise& noise, int width, int height, float baseFrequency, int numLayers, float frequencyMult, float amplitudeMult);
std::vector<float> generateWoodNoise(const ValueNoise& noise, int width, int height, float frequency, float ringFrequency);

std::vector<float> generateValueNoise(const ValueNoise& noise, int width, int height, float frequency) {
    std::vector<float> valueNoise(width * height);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            // Convert pixel coordinates to noise space based on the frequency
            Vec2f point((i / static_cast<float>(width)) * frequency,
                        (j / static_cast<float>(height)) * frequency);
            // Evaluate the noise function at this point
            float noiseValue = noise.eval(point);
            // Store the result in the vector
            valueNoise[j * width + i] = noiseValue;
        }
    }
    return valueNoise;
}

std::vector<float> generateFractalNoise(const ValueNoise& noise, int width, int height, float baseFrequency, int numLayers, float frequencyMult, float amplitudeMult) {
    std::vector<float> fractalNoise(width * height);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float frequency = baseFrequency;
            float amplitude = 1.0f;
            float value = 0.0f;
            for (int layer = 0; layer < numLayers; ++layer) {
                Vec2f p(i * frequency / width, j * frequency / height);
                value += noise.eval(p) * amplitude;
                frequency *= frequencyMult;
                amplitude *= amplitudeMult;
            }
            fractalNoise[j * width + i] = value;
        }
    }
    return fractalNoise;
}


std::vector<float> generateTurbulenceNoise(const ValueNoise& noise, int width, int height, float baseFrequency, int numLayers, float frequencyMult, float amplitudeMult) {
    std::vector<float> turbulenceNoise(width * height);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float frequency = baseFrequency;
            float amplitude = 1.0f;
            float value = 0.0f;
            for (int layer = 0; layer < numLayers; ++layer) {
                Vec2f p(i * frequency / width, j * frequency / height);
                value += std::fabs(2.0f * noise.eval(p) - 1.0f) * amplitude;
                frequency *= frequencyMult;
                amplitude *= amplitudeMult;
            }
            turbulenceNoise[j * width + i] = value;
        }
    }
    return turbulenceNoise;
}

std::vector<float> generateMarbleNoise(const ValueNoise& noise, int width, int height, float baseFrequency, int numLayers, float frequencyMult, float amplitudeMult) {
    std::vector<float> marbleNoise(width * height);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float x = i / static_cast<float>(width);
            float y = j / static_cast<float>(height);
            float noiseValue = 0.0f;
            float frequency = baseFrequency;
            float amplitude = 1.0f;
            for (int layer = 0; layer < numLayers; ++layer) {
                Vec2f p(x * frequency, y * frequency);
                noiseValue += noise.eval(p) * amplitude;
                frequency *= frequencyMult;
                amplitude *= amplitudeMult;
            }
            float value = 0.5f * sin(8.0f * x + noiseValue * 10.0f) + 0.5f;
            marbleNoise[j * width + i] = value;
        }
    }
    return marbleNoise;
}


std::vector<float> generateWoodNoise(const ValueNoise& noise, int width, int height, float frequency, float ringFrequency) {
    std::vector<float> woodNoise(width * height);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            Vec2f p(i * frequency / width, j * frequency / height);
            float value = noise.eval(p);
            value = value * ringFrequency - static_cast<int>(value * ringFrequency);
            woodNoise[j * width + i] = value;
        }
    }
    return woodNoise;
}
int main() {
    const unsigned imageWidth = 512, imageHeight = 512;
    ValueNoise noise;

    // Generate and save Value Noise
    auto valueNoiseMap = generateValueNoise(noise, imageWidth, imageHeight, 0.05f);
    savePPM("value_noise.ppm", valueNoiseMap, imageWidth, imageHeight);

    // Generate and save Fractal Noise
    auto fractalNoiseMap = generateFractalNoise(noise, imageWidth, imageHeight, 0.02f, 5, 1.8f, 0.5f);
    savePPM("fractal_noise.ppm", fractalNoiseMap, imageWidth, imageHeight);

    // Generate and save Turbulence Noise
    auto turbulenceNoiseMap = generateTurbulenceNoise(noise, imageWidth, imageHeight, 0.02f, 5, 1.8f, 0.5f);
    savePPM("turbulence_noise.ppm", turbulenceNoiseMap, imageWidth, imageHeight);

    // Generate and save Marble Noise
    auto marbleNoiseMap = generateMarbleNoise(noise, imageWidth, imageHeight, 0.02f, 5, 1.8f, 0.5f);
    savePPM("marble_noise.ppm", marbleNoiseMap, imageWidth, imageHeight);

    // Generate and save Wood Noise
    auto woodNoiseMap = generateWoodNoise(noise, imageWidth, imageHeight, 0.05f, 10.0f);
    savePPM("wood_noise.ppm", woodNoiseMap, imageWidth, imageHeight);

    std::cout << "All patterns have been generated and saved." << std::endl;

    return 0;
}