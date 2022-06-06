//
// Created by goksu on 2/25/20.
//

#include <fstream>
#include "Scene.hpp"
#include "Renderer.hpp"
#include "omp.h"
#include "ThreadPool.h"


inline float deg2rad(const float& deg) { return deg * M_PI / 180.0; }

const float EPSILON = 0.00001;

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene)
{
    std::vector<Vector3f> framebuffer(scene.width * scene.height);

    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = scene.width / (float)scene.height;
    Vector3f eye_pos(278, 273, -800);
    int m = 0;

    bool multithread = true;

    if(!multithread)
    {
    // change the spp value to change sample ammount
    int spp = 512;
    std::cout << "SPP: " << spp << "\n";
    //#pragma omp parallel for 
    //config for the simple multithreading code
    for (uint32_t j = 0; j < scene.height; ++j) {
        for (uint32_t i = 0; i < scene.width; ++i) {
            // generate primary ray direction
            float x = (2 * (i + 0.5) / (float)scene.width - 1) *
                      imageAspectRatio * scale;
            float y = (1 - 2 * (j + 0.5) / (float)scene.height) * scale;

            Vector3f dir = normalize(Vector3f(-x, y, 1));
            thread_local Vector3f color = Vector3f(0.0);
            for (int k = 0; k < spp; k++){
                framebuffer[m] += scene.castRay(Ray(eye_pos, dir), 0) / spp;  
            }
            m++;
            }
            UpdateProgress(j/(float)scene.height);
        }
    UpdateProgress(1.f);
    }
    else
    {
    float total = (float)scene.width * (float)scene.height;
    // change the spp value to change sample ammount
    int spp = 32;
    std::cout << "SPP: " << spp << "\n";

    ThreadPool pool(std::thread::hardware_concurrency());
    pool.init();
    std::mutex l;

    //config for the simple multithreading code
    auto rayTracingPixel = [&](int i, int j) {
        // generate primary ray direction
        float x = (2 * (i + 0.5) / (float)scene.width - 1) *
            imageAspectRatio * scale;
        float y = (1 - 2 * (j + 0.5) / (float)scene.height) * scale;

        Vector3f dir = normalize(Vector3f(-x, y, 1));
        thread_local Vector3f color;
        color = Vector3f(0);
        for (int k = 0; k < spp; k++) {
            color += scene.castRay(Ray(eye_pos, dir), 0) / spp;
        }
        framebuffer[j * scene.width + i] += color;
        l.lock();
        m++;
        std::cout << m << "\r";
    };

    for (uint32_t j = 0; j < scene.height; ++j) {
        for (uint32_t i = 0; i < scene.width; ++i) {
            pool.submit(rayTracingPixel, i, j);
        }
    }
    pool.myShutdown();

    UpdateProgress(1.f);
    }


    // save framebuffer to file
    FILE* fp = fopen("binary.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);    
}