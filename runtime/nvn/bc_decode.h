#pragma once
#include <stdlib.h>
#include <string.h>
#include "tegra.h"
#define BCDEC_STATIC
#define BCDEC_IMPLEMENTATION
#define BCDEC_BC4BC5_PRECISE
#include "bcdec.h"

// Decode into the destination's exact texel dimensions, including small/odd edge mips and volume slices.
static uint8_t *decode_bc_level(const uint8_t *src, uint32_t format, LevelLayout l, uint32_t bytes, bool bc5_unorm) {
    uint8_t *out = malloc((uint64_t)l.w * l.h * l.d * bytes);
    if (!out) return NULL;
    uint32_t block_bytes = nvn_format_info(format)->bpb;
    for (uint32_t z = 0; z < l.d; z++) for (uint32_t y = 0; y < l.hb; y++) for (uint32_t x = 0; x < l.wb; x++) {
        const uint8_t *block = src + (((uint64_t)z * l.hb + y) * l.wb + x) * block_bytes;
        uint64_t rgba[16] = {0};  // aligned scratch for at most sixteen RGBA16F pixels
        switch (format) {
        case 0x41: case 0x42: case 0x45: case 0x46: bcdec_bc1(block, rgba, 16); break;
        case 0x43: case 0x47: bcdec_bc2(block, rgba, 16); break;
        case 0x44: case 0x48: bcdec_bc3(block, rgba, 16); break;
        case 0x49: case 0x4A: bcdec_bc4(block, rgba, 4, format == 0x4A); break;
        case 0x4B: case 0x4C: bcdec_bc5(block, rgba, 8, format == 0x4C && !bc5_unorm); break;
        case 0x4D: case 0x4E: bcdec_bc7(block, rgba, 16); break;
        case 0x4F: case 0x50: {
            uint16_t rgb[48];
            bcdec_bc6h_half(block, rgb, 12, format == 0x4F);
            for (int i = 0; i < 16; i++) {
                memcpy((uint8_t *)rgba + i * 8, rgb + i * 3, 6);
                ((uint16_t *)rgba)[i * 4 + 3] = 0x3c00;  // alpha 1.0
            }
            break;
        }
        }
        uint32_t width = (l.w - x * 4 < 4 ? l.w - x * 4 : 4);
        for (uint32_t row = 0; row < 4 && y * 4 + row < l.h; row++)
            memcpy(out + (((uint64_t)z * l.h + y * 4 + row) * l.w + x * 4) * bytes,
                (uint8_t *)rgba + row * 4 * bytes, width * bytes);
    }
    return out;
}
