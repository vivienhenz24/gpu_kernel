#include <metal_stdlib>
using namespace metal;

constant half kWeights[25] = {
    0.0183542h, 0.0219671h, 0.0258836h, 0.0300255h, 0.0342902h,
    0.0385535h, 0.0426749h, 0.0465044h, 0.0498920h, 0.0526964h,
    0.0547956h, 0.0560950h, 0.0565350h, 0.0560950h, 0.0547956h,
    0.0526964h, 0.0498920h, 0.0465044h, 0.0426749h, 0.0385535h,
    0.0342902h, 0.0300255h, 0.0258836h, 0.0219671h, 0.0183542h
};

kernel void gaussian_blur_h(
    texture2d<half, access::read> in_tex [[texture(0)]],
    texture2d<half, access::write> out_tex [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = in_tex.get_width();
    uint height = in_tex.get_height();
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    half4 sum = half4(0.0h);
    int x = int(gid.x);
    int y = int(gid.y);
    int max_x = int(width) - 1;

    for (int i = -12; i <= 12; ++i) {
        int sx = clamp(x + i, 0, max_x);
        sum += in_tex.read(uint2(sx, y)) * kWeights[i + 12];
    }

    out_tex.write(sum, gid);
}

kernel void gaussian_blur_v(
    texture2d<half, access::read> in_tex [[texture(0)]],
    texture2d<half, access::write> out_tex [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = in_tex.get_width();
    uint height = in_tex.get_height();
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    half4 sum = half4(0.0h);
    int x = int(gid.x);
    int y = int(gid.y);
    int max_y = int(height) - 1;

    for (int i = -12; i <= 12; ++i) {
        int sy = clamp(y + i, 0, max_y);
        sum += in_tex.read(uint2(x, sy)) * kWeights[i + 12];
    }

    out_tex.write(sum, gid);
}
