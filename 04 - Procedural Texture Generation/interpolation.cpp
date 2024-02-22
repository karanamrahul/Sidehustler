#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>



struct Vec3f {
    float x , y , z;

    Vec3f() : x(0) , y(0) , z(0) {}
    Vec3f(float x , float y , float z) : x(x) , y(y) , z(z) {}

    Vec3f operator+(const Vec3f& other) const {
        return Vec3f(x + other.x , y + other.y , z + other.z);
    }

    Vec3f operator*(float scalar) const {
            return Vec3f(x * scalar , y* scalar ,z * scalar);
        }
};


Vec3f bilinear(float tx , float ty, const Vec3f &c00 , const Vec3f &c10,const Vec3f &c01 , const Vec3f &c11){
    Vec3f a = c00 * (1 -tx) + c10 * tx;
    Vec3f b = c01 * (1 -ty) + c11 * tx;
    return a*(1-ty) + b * ty;

}


Vec3f trilinear(float tx, float ty, float tz, const Vec3f &c000, const Vec3f &c100, const Vec3f &c010, const Vec3f &c110, const Vec3f &c001, const Vec3f &c101, const Vec3f &c011, const Vec3f &c111) {
    Vec3f c00 = c000 * (1 - tx) + c100 * tx;
    Vec3f c10 = c010 * (1 - tx) + c110 * tx;
    Vec3f c01 = c001 * (1 - tx) + c101 * tx;
    Vec3f c11 = c011 * (1 - tx) + c111 * tx;

    Vec3f c0 = c00 * (1 - ty) + c10 * ty;
    Vec3f c1 = c01 * (1 - ty) + c11 * ty;

    return c0 * (1 - tz) + c1 * tz;
}



void saveToPPM(const char* filename,Vec3f* data,int width,int height){
     std::ofstream ofs(filename, std::ios::binary); // Open the file in binary mode
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // write the ppm header

    ofs << "P6\n" << width << " " << height << "\n255\n";

    // write the pixel data

    for(int i = 0; i < width * height; ++i) {

        // convert each Vec3f component from [0,1] to [0,255]
        unsigned char r = static_cast<unsigned char>(std::min(1.0f, data[i].x) * 255.0f);
        unsigned char g = static_cast<unsigned char>(std::min(1.0f, data[i].y) * 255.0f);
        unsigned char b = static_cast<unsigned char>(std::min(1.0f, data[i].z) * 255.0f);

        // write the color to the file

        ofs.write(reinterpret_cast<char*>(&r),sizeof(r));
        ofs.write(reinterpret_cast<char*>(&g),sizeof(g));
        ofs.write(reinterpret_cast<char*>(&b),sizeof(b));

    }
    ofs.close();
    std::cout << "File saved: " << filename << std::endl;

}



void testBilinearInterpolation() {

    // seed the random number generator

    srand(static_cast<unsigned int>(time(0))); // random number seed

    // testing bilinear interpolation
    int imageWidth = 512;
    int gridSizeX = 9, gridSizeY = 9;
    Vec3f *grid2d = new Vec3f[(gridSizeX + 1) * (gridSizeY + 1)]; // lattices
    // fill the grid with random colors
    for (int j = 0, k = 0; j <= gridSizeY; ++j) {
        for (int i = 0; i <= gridSizeX; ++i, ++k) {
            grid2d[j * (gridSizeX + 1) + i] = Vec3f(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);        }
    }
    // now compute our final image using bilinear interpolation
    Vec3f *imageData = new Vec3f[imageWidth*imageWidth], *pixel = imageData;
    for (int j = 0; j < imageWidth; ++j) {
        for (int i = 0; i < imageWidth; ++i) {
            // convert i,j to grid coordinates
            auto gx = i / float(imageWidth) * gridSizeX;
            auto gy = j / float(imageWidth) * gridSizeY;
            int gxi = int(gx);
            int gyi = int(gy);
            const Vec3f & c00 = grid2d[gyi * (gridSizeX + 1) + gxi];
            const Vec3f & c10 = grid2d[gyi * (gridSizeX + 1) + (gxi + 1)];
            const Vec3f & c01 = grid2d[(gyi + 1) * (gridSizeX + 1) + gxi];
            const Vec3f & c11 = grid2d[(gyi + 1) * (gridSizeX + 1) + (gxi + 1)];
            *(pixel++) = bilinear(gx - gxi, gy - gyi, c00, c10, c01, c11);
        }
    }
    saveToPPM("./bilinear.ppm", imageData, imageWidth, imageWidth);
    delete [] imageData;
    delete grid2d;
}



void testTrilinearInterpolation() {
    srand(static_cast<unsigned int>(time(0))); // Seed for random color generation

    int imageWidth = 512, imageDepth = 512; // For simplicity, we use a square image and depth
    int gridSizeX = 9, gridSizeY = 9, gridSizeZ = 9; // Define the size of the grid
    Vec3f *grid3d = new Vec3f[(gridSizeX + 1) * (gridSizeY + 1) * (gridSizeZ + 1)]; // 3D grid

    // Fill the grid with random colors
    for (int z = 0; z <= gridSizeZ; ++z) {
        for (int y = 0; y <= gridSizeY; ++y) {
            for (int x = 0; x <= gridSizeX; ++x) {
                grid3d[z * (gridSizeX + 1) * (gridSizeY + 1) + y * (gridSizeX + 1) + x] = Vec3f(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
            }
        }
    }

    // Compute the final image using trilinear interpolation
    Vec3f *imageData = new Vec3f[imageWidth * imageWidth];
    for (int j = 0; j < imageWidth; ++j) {
        for (int i = 0; i < imageWidth; ++i) {
            float gx = i / float(imageWidth) * gridSizeX;
            float gy = j / float(imageWidth) * gridSizeY;
            float gz = gridSizeZ / 2.0f; // For demonstration, we use the middle layer in Z
            int gxi = int(gx), gyi = int(gy), gzi = int(gz);

            // Retrieve the corners of the cell in 3D
            const Vec3f &c000 = grid3d[gzi * (gridSizeX + 1) * (gridSizeY + 1) + gyi * (gridSizeX + 1) + gxi];
            const Vec3f &c100 = grid3d[gzi * (gridSizeX + 1) * (gridSizeY + 1) + gyi * (gridSizeX + 1) + (gxi + 1)];
            const Vec3f &c010 = grid3d[gzi * (gridSizeX + 1) * (gridSizeY + 1) + (gyi + 1) * (gridSizeX + 1) + gxi];
            const Vec3f &c110 = grid3d[gzi * (gridSizeX + 1) * (gridSizeY + 1) + (gyi + 1) * (gridSizeX + 1) + (gxi + 1)];
            const Vec3f &c001 = grid3d[(gzi + 1) * (gridSizeX + 1) * (gridSizeY + 1) + gyi * (gridSizeX + 1) + gxi];
            const Vec3f &c101 = grid3d[(gzi + 1) * (gridSizeX + 1) * (gridSizeY + 1) + gyi * (gridSizeX + 1) + (gxi + 1)];
            const Vec3f &c011 = grid3d[(gzi + 1) * (gridSizeX + 1) * (gridSizeY + 1) + (gyi + 1) * (gridSizeX + 1) + gxi];
            const Vec3f &c111 = grid3d[(gzi + 1) * (gridSizeX + 1) * (gridSizeY + 1) + (gyi + 1) * (gridSizeX + 1) + (gxi + 1)];

            // Calculate the interpolated color
            imageData[j * imageWidth + i] = trilinear(gx - gxi, gy - gyi, gz - gzi, c000, c100, c010, c110, c001, c101, c011, c111);
        }
    }

    // Save the 2D cross-section of the 3D texture as a PPM file
    saveToPPM("./trilinear.ppm", imageData, imageWidth, imageWidth);

    delete[] imageData;
    delete[] grid3d;
}


int main(){
    testBilinearInterpolation();
    testTrilinearInterpolation();
    return 0;
}