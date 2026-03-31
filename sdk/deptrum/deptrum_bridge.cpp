/*
 * deptrum_bridge.cpp – thin C bridge over the Deptrum C++ stream SDK
 *
 * Exposes plain-C functions so Python ctypes can drive an Aurora 930 camera
 * without any C++ ABI dependency.
 *
 * Build:
 *   g++ -shared -fPIC -O2 -std=c++17 \
 *       -I include \
 *       -L lib -ldeptrum_stream_aurora900 \
 *       -Wl,-rpath,'$ORIGIN/lib' \
 *       -o lib/libdeptrum_bridge.so deptrum_bridge.cpp
 */

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "deptrum/device.h"
#include "deptrum/stream.h"
#include "deptrum/aurora900_series.h"

using namespace deptrum;
using namespace deptrum::stream;

/* ── internal state ─────────────────────────────────────────────── */
static std::shared_ptr<DeviceManager> g_mgr;
static std::shared_ptr<Device>        g_dev;
static Stream*                        g_stream  = nullptr;
static std::atomic<bool>              g_opened{false};
static std::atomic<bool>              g_streaming{false};

static int g_rgb_w = 0, g_rgb_h = 0;
static int g_depth_w = 0, g_depth_h = 0;
static int g_ir_w = 0, g_ir_h = 0;

/* ── C API ──────────────────────────────────────────────────────── */
extern "C" {

int deptrum_device_count() {
    auto mgr = DeviceManager::GetInstance();
    if (!mgr) return -1;
    std::vector<DeviceInformation> list;
    int st = mgr->GetDeviceList(list);
    if (st != 0) return -1;
    return static_cast<int>(list.size());
}

int deptrum_open(int resolution_mode_index,
                 int rgb_enable, int ir_enable, int depth_enable,
                 int point_cloud_enable) {
    if (g_opened) return 0;

    g_mgr = DeviceManager::GetInstance();
    if (!g_mgr) return -1;

    std::vector<DeviceInformation> list;
    int st = g_mgr->GetDeviceList(list);
    if (st != 0 || list.empty()) return -2;

    g_dev = g_mgr->CreateDevice(list[0]);
    if (!g_dev) return -3;

    st = g_dev->Open();
    if (st != 0) { g_dev.reset(); return -4; }

    /* pick resolution from supported modes */
    std::vector<std::tuple<FrameMode,FrameMode,FrameMode>> modes;
    g_dev->GetSupportedFrameMode(modes);
    if (modes.empty()) { g_dev->Close(); g_dev.reset(); return -5; }
    if (resolution_mode_index < 0 || resolution_mode_index >= (int)modes.size())
        resolution_mode_index = 0;

    auto [ir_mode, rgb_mode, depth_mode] = modes[resolution_mode_index];
    st = g_dev->SetMode(ir_mode, rgb_mode, depth_mode);
    if (st != 0) { g_dev->Close(); g_dev.reset(); return -6; }

    /* Aurora 930-specific settings */
    auto a9 = std::dynamic_pointer_cast<Aurora900>(g_dev);
    if (a9) {
        a9->SwitchAutoExposure(true);
        a9->SwitchAlignedMode(true);
        a9->DepthCorrection(true);
        a9->FilterOutRangeDepthMap(150, 4000);
        a9->SetRemoveFilterSize(110);
        a9->SetLaserDriver(1.0f);
        a9->SetIrFps(12);
    }

    /* build stream types */
    std::vector<StreamType> types;
    if (rgb_enable)         types.push_back(StreamType::kRgb);
    if (depth_enable || ir_enable)
                            types.push_back(StreamType::kDepthIr);
    if (point_cloud_enable) types.push_back(StreamType::kPointCloud);
    if (types.empty())      types.push_back(StreamType::kRgb);

    st = g_dev->CreateStream(g_stream, types);
    if (st != 0) { g_dev->Close(); g_dev.reset(); return -7; }

    g_opened = true;
    return 0;
}

int deptrum_start() {
    if (!g_opened || !g_stream) return -1;
    if (g_streaming) return 0;
    int st = g_stream->Start();
    if (st != 0) return st;
    g_streaming = true;
    return 0;
}

/*
 * Grab one set of frames.  Caller supplies pre-allocated buffers.
 * Pass NULL for any buffer you don't need.
 *
 *   rgb_buf     – at least (w*h*3) bytes, receives BGR888 data
 *   depth_buf   – at least (w*h*2) bytes, receives 16-bit depth in mm
 *   ir_buf      – at least (w*h) bytes, receives 8-bit IR
 *
 * On success the actual width/height are written to *_w / *_h.
 * Returns 0 on success.
 */
int deptrum_grab(unsigned char* rgb_buf,   int* rgb_w,   int* rgb_h,
                 unsigned char* depth_buf, int* depth_w, int* depth_h,
                 unsigned char* ir_buf,    int* ir_w,    int* ir_h,
                 int timeout_ms) {
    if (!g_streaming || !g_stream) return -1;

    StreamFrames frames;
    int st = g_stream->GetFrames(frames, timeout_ms > 0 ? timeout_ms : 2000);
    if (st != 0) return st;

    for (int i = 0; i < frames.count; i++) {
        auto& f = *frames.frame_ptr[i];

        if (f.frame_type == FrameType::kRgbFrame && rgb_buf) {
            int w = f.cols, h = f.rows;
            int expected_nv12 = static_cast<int>(w * h * 1.5f);
            if (f.size == expected_nv12) {
                /* NV12 → stored as-is; Python side will convert with OpenCV */
                if (rgb_w) *rgb_w = w;
                if (rgb_h) *rgb_h = h;
                /* store a flag: negative height means NV12 */
                if (rgb_h) *rgb_h = -h;
                std::memcpy(rgb_buf, f.data.get(), f.size);
            } else {
                /* BGR888 */
                if (rgb_w) *rgb_w = w;
                if (rgb_h) *rgb_h = h;
                std::memcpy(rgb_buf, f.data.get(), w * h * 3);
            }
            g_rgb_w = w;
            g_rgb_h = h;
        }

        if (f.frame_type == FrameType::kDepthFrame && depth_buf) {
            int w = f.cols, h = f.rows;
            if (depth_w) *depth_w = w;
            if (depth_h) *depth_h = h;
            std::memcpy(depth_buf, f.data.get(), w * h * 2);
            g_depth_w = w;
            g_depth_h = h;
        }

        if (f.frame_type == FrameType::kIrFrame && ir_buf) {
            int w = f.cols, h = f.rows;
            if (ir_w) *ir_w = w;
            if (ir_h) *ir_h = h;
            int bpp = f.bits_per_pixel;
            if (bpp <= 8) {
                std::memcpy(ir_buf, f.data.get(), w * h);
            } else {
                std::memcpy(ir_buf, f.data.get(), w * h * 2);
            }
            g_ir_w = w;
            g_ir_h = h;
        }
    }
    return 0;
}

int deptrum_get_frame_size(int* rgb_w, int* rgb_h,
                           int* depth_w, int* depth_h,
                           int* ir_w, int* ir_h) {
    if (rgb_w)   *rgb_w   = g_rgb_w;
    if (rgb_h)   *rgb_h   = g_rgb_h;
    if (depth_w) *depth_w = g_depth_w;
    if (depth_h) *depth_h = g_depth_h;
    if (ir_w)    *ir_w    = g_ir_w;
    if (ir_h)    *ir_h    = g_ir_h;
    return 0;
}

int deptrum_stop() {
    if (!g_streaming) return 0;
    g_streaming = false;
    if (g_stream) g_stream->Stop();
    return 0;
}

int deptrum_close() {
    deptrum_stop();
    if (!g_opened) return 0;
    g_opened = false;
    if (g_stream && g_dev) {
        g_dev->DestroyStream(g_stream);
        g_stream = nullptr;
    }
    if (g_dev) {
        g_dev->Close();
        g_dev.reset();
    }
    g_mgr.reset();
    return 0;
}

const char* deptrum_sdk_version() {
    static char buf[64] = {0};
    if (buf[0] == 0) {
        auto mgr = DeviceManager::GetInstance();
        if (mgr) {
            std::vector<DeviceInformation> list;
            mgr->GetDeviceList(list);
            if (!list.empty()) {
                auto dev = mgr->CreateDevice(list[0]);
                if (dev) {
                    std::string v = dev->GetSdkVersion();
                    std::strncpy(buf, v.c_str(), sizeof(buf) - 1);
                }
            }
        }
        if (buf[0] == 0) std::strncpy(buf, "unknown", sizeof(buf) - 1);
    }
    return buf;
}

} /* extern "C" */
