//#include "ospray/ospray.h"

#include "ospray/ospray_cpp.h"
#include "ospray/ospray_cpp/ext/rkcommon.h"
#include "rkcommon/utility/SaveImage.h"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace rkcommon::math;


inline void initializeOSPRay(
        int argc, const char **argv, bool errorsFatal = true) {

// image size
    vec2i imgSize = {1024, 720};

// camera
    vec3d cam_pos{0.f, 0.f, 0.f};
    vec3d cam_up{0.f, 1.f, 0.f};
    vec3d cam_view{0.1f, 0.f, 1.f};

// triangle mesh data
    std::vector<vec3f> vertex = {vec3f(-1.0f, -1.0f, 3.0f),
                                 vec3f(-1.0f, 1.0f, 3.0f),
                                 vec3f(1.0f, -1.0f, 3.0f),
                                 vec3f(0.1f, 0.1f, 0.3f)};

    std::vector<vec4f> color = {vec4f(0.9f, 0.0f, 0.0f, 1.0f),
                                vec4f(0.0f, 0.8f, 0.0f, 1.0f),
                                vec4f(0.0f, 0.8f, 0.0f, 1.0f),
                                vec4f(0.0f, 0.0f, 0.5f, 1.0f)};

    std::vector<vec3ui> index = {vec3ui(0, 1, 2), vec3ui(1, 2, 3)};

    // initialize OSPRay; OSPRay parses (and removes) its commandline parameters,
    // e.g. "--osp:debug"
    OSPError initError = ospInit(&argc, argv);

    if (initError != OSP_NO_ERROR)
        throw std::runtime_error("OSPRay not initialized correctly!");

    OSPDevice device = ospGetCurrentDevice();
    if (!device)
        throw std::runtime_error("OSPRay device could not be fetched!");

    // use scoped lifetimes of wrappers to release everything before ospShutdown()
    {
        // create and setup camera
        ospray::cpp::Camera camera("perspective");
        camera.setParam("aspect", imgSize.x / (float)imgSize.y);
        camera.setParam("position", cam_pos);
        camera.setParam("direction", cam_view);
        camera.setParam("up", cam_up);
        camera.commit(); // commit each object to indicate modifications are done

        // create and setup model and mesh
        ospray::cpp::Geometry mesh("mesh");
        mesh.setParam("vertex.position", ospray::cpp::CopiedData(vertex));
        mesh.setParam("vertex.color", ospray::cpp::CopiedData(color));
        mesh.setParam("index", ospray::cpp::CopiedData(index));
        mesh.commit();

        // put the mesh into a model
        ospray::cpp::GeometricModel model(mesh);
        model.commit();

        // put the model into a group (collection of models)
        ospray::cpp::Group group;
        group.setParam("geometry", ospray::cpp::CopiedData(model));
        group.commit();

        // put the group into an instance (give the group a world transform)
        ospray::cpp::Instance instance(group);
        instance.commit();

        // put the instance in the world
        ospray::cpp::World world;
        world.setParam("instance", ospray::cpp::CopiedData(instance));

        // create and setup light for Ambient Occlusion
        ospray::cpp::Light light("ambient");
        light.commit();

        world.setParam("light", ospray::cpp::CopiedData(light));
        world.commit();

        // create renderer, choose Scientific Visualization renderer
        ospray::cpp::Renderer renderer("scivis");

        // complete setup of renderer
        renderer.setParam("aoSamples", 1);
        renderer.setParam("backgroundColor", 0.0f); // white, transparent
        renderer.commit();

        // create and setup framebuffer
        ospray::cpp::FrameBuffer framebuffer(
                imgSize.x, imgSize.y, OSP_FB_SRGBA, OSP_FB_COLOR | OSP_FB_ACCUM);
        framebuffer.clear();

        // render one frame
        framebuffer.renderFrame(renderer, camera, world);

        // access framebuffer and write its content as PPM file
        uint32_t *fb = (uint32_t *)framebuffer.map(OSP_FB_COLOR);
        rkcommon::utility::writePPM("firstFrameCpp.ppm", imgSize.x, imgSize.y, fb);
        framebuffer.unmap(fb);
        std::cout << "rendering initial frame to firstFrameCpp.ppm" << std::endl;

        // render 10 more frames, which are accumulated to result in a better
        // converged image
        for (int frames = 0; frames < 100; frames++)
            framebuffer.renderFrame(renderer, camera, world);

        fb = (uint32_t *)framebuffer.map(OSP_FB_COLOR);
        rkcommon::utility::writePPM(
                "accumulatedFrameCpp.ppm", imgSize.x, imgSize.y, fb);
        framebuffer.unmap(fb);
        std::cout << "rendering 10 accumulated frames to accumulatedFrameCpp.ppm"
                  << std::endl;

        ospray::cpp::PickResult res =
                framebuffer.pick(renderer, camera, world, 0.5f, 0.5f);

        if (res.hasHit) {
            std::cout << "picked geometry [instance: " << res.instance.handle()
                      << ", model: " << res.model.handle()
                      << ", primitive: " << res.primID << "]" << std::endl;
        }
    }

    // set an error callback to catch any OSPRay errors and exit the application
    if (errorsFatal) {
        ospDeviceSetErrorCallback(
                device,
                [](void *, OSPError error, const char *errorDetails) {
                    std::cerr << "OSPRay error: " << errorDetails << std::endl;
                    exit(error);
                },
                nullptr);
    } else {
        ospDeviceSetErrorCallback(
                device,
                [](void *, OSPError, const char *errorDetails) {
                    std::cerr << "OSPRay error: " << errorDetails << std::endl;
                },
                nullptr);
    }

    ospDeviceSetStatusCallback(
            device, [](void *, const char *msg) { std::cout << msg; }, nullptr);

    bool warnAsErrors = true;
    auto logLevel = OSP_LOG_WARNING;

    ospDeviceSetParam(device, "warnAsError", OSP_BOOL, &warnAsErrors);
    ospDeviceSetParam(device, "logLevel", OSP_INT, &logLevel);

    ospDeviceCommit(device);
    ospDeviceRelease(device);
}

int main(int argc, const char *argv[]) {
    initializeOSPRay(argc, argv, false);

    ospShutdown();
    return 0;
}
