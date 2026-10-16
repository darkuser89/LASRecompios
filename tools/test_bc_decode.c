#include <assert.h>
#include <stdio.h>
#include "../runtime/nvn/bc_decode.h"

static void solid(uint32_t format, const uint64_t *block, uint32_t bytes, const uint8_t *expected) {
    const FormatInfo *fi = nvn_format_info(format);
    // Non-block-aligned mips and multiple volume slices exercise edge clipping and source/destination strides.
    for (uint32_t w = 1; w <= 7; w++) for (uint32_t h = 1; h <= 5; h++) {
        LevelLayout level = {.w=w,.h=h,.d=2,.wb=(w+3)/4,.hb=(h+3)/4};
        uint8_t *packed = malloc(level.wb * level.hb * level.d * fi->bpb);
        for (uint32_t i=0; i<level.wb*level.hb*level.d; i++) memcpy(packed+i*fi->bpb,block,fi->bpb);
        uint8_t *pixels = decode_bc_level(packed,format,level,bytes,false);
        assert(pixels);
        for (uint32_t i=0; i<w*h*level.d; i++) assert(!memcmp(pixels+i*bytes,expected,bytes));
        free(pixels); free(packed);
    }
}

int main(void) {
    const uint8_t red[] = {255,0,0,255}, transparent[] = {0,0,0,0};
    uint64_t bc1[] = {0x0000000007e0f800ull};
    solid(0x42,bc1,4,red);
    solid(0x46,bc1,4,red);  // sRGB bytes remain encoded; Metal performs the conversion when sampled.
    uint64_t alpha[] = {0xffffffffFFFF0000ull};
    solid(0x42,alpha,4,transparent);
    uint64_t bc2[] = {~0ull,bc1[0]};
    solid(0x43,bc2,4,red);
    uint64_t bc3[] = {255,bc1[0]};
    solid(0x44,bc3,4,red);
    uint64_t bc4[] = {0x7f81};
    const uint8_t snorm[] = {129};
    solid(0x4a,bc4,1,snorm);
    uint64_t bc5[] = {0x7f81,63};
    const uint8_t rg[] = {129,63};
    solid(0x4c,bc5,2,rg);
    uint64_t black[] = {0,0};
    solid(0x4d,black,4,transparent);  // invalid BC7 mode has a defined transparent-black fallback.
    const uint8_t black_half[] = {0,0,0,0,0,0,0,0x3c};
    solid(0x50,black,8,black_half);
    puts("BC1/2/3/4/5/6H/7: known colours, signed channels, sRGB bytes, edge mips and volume slices passed");
}
