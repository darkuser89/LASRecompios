// Only suppress identical state within one encoder. Clears bypass these setters and invalidate it.
#pragma once
#import <Metal/Metal.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    __weak id<MTLRenderCommandEncoder> encoder;
    __unsafe_unretained id<MTLRenderPipelineState> pipeline;
    __unsafe_unretained id<MTLDepthStencilState> depth;
    uint32_t valid;
    uint32_t stencil[2];
    float blend[4], bias[3];
    MTLCullMode cull;
    MTLWinding winding;
    MTLTriangleFillMode fill;
    MTLViewport viewport;
    MTLScissorRect scissor;
    uint64_t requests, calls;
    bool legacy;
} RenderState;

static inline void render_state_reset(RenderState *s) { s->valid = 0; s->encoder = nil; }
static inline void render_state_begin(RenderState *s, id<MTLRenderCommandEncoder> encoder) {
    if (s->encoder != encoder) { s->valid = 0; s->encoder = encoder; }
}

#define RENDER_STATE_SCALAR(name, type, field, bit, method) \
static inline void render_state_##name(RenderState *s, id<MTLRenderCommandEncoder> e, type v) { \
    s->requests++; \
    if (s->legacy || !(s->valid & (1u << bit)) || s->field != v) { \
        [e method:v]; s->field = v; s->valid |= 1u << bit; s->calls++; \
    } \
}
RENDER_STATE_SCALAR(pipeline, id<MTLRenderPipelineState>, pipeline, 0, setRenderPipelineState)
RENDER_STATE_SCALAR(depth, id<MTLDepthStencilState>, depth, 1, setDepthStencilState)
RENDER_STATE_SCALAR(cull, MTLCullMode, cull, 2, setCullMode)
RENDER_STATE_SCALAR(winding, MTLWinding, winding, 3, setFrontFacingWinding)
RENDER_STATE_SCALAR(fill, MTLTriangleFillMode, fill, 4, setTriangleFillMode)
#undef RENDER_STATE_SCALAR

static inline void render_state_stencil(RenderState *s, id<MTLRenderCommandEncoder> e, uint32_t front, uint32_t back) {
    s->requests++;
    if (s->legacy || !(s->valid & 32) || s->stencil[0] != front || s->stencil[1] != back) {
        [e setStencilFrontReferenceValue:front backReferenceValue:back];
        s->stencil[0] = front; s->stencil[1] = back; s->valid |= 32; s->calls++;
    }
}
static inline void render_state_blend(RenderState *s, id<MTLRenderCommandEncoder> e, const float v[4]) {
    s->requests++;
    if (s->legacy || !(s->valid & 64) || memcmp(s->blend, v, sizeof s->blend)) {
        [e setBlendColorRed:v[0] green:v[1] blue:v[2] alpha:v[3]];
        memcpy(s->blend, v, sizeof s->blend); s->valid |= 64; s->calls++;
    }
}
static inline void render_state_bias(RenderState *s, id<MTLRenderCommandEncoder> e, const float v[3]) {
    s->requests++;
    if (s->legacy || !(s->valid & 128) || memcmp(s->bias, v, sizeof s->bias)) {
        [e setDepthBias:v[1] slopeScale:v[0] clamp:v[2]];
        memcpy(s->bias, v, sizeof s->bias); s->valid |= 128; s->calls++;
    }
}
static inline void render_state_viewport(RenderState *s, id<MTLRenderCommandEncoder> e, MTLViewport v) {
    s->requests++;
    if (s->legacy || !(s->valid & 256) || memcmp(&s->viewport, &v, sizeof v)) {
        [e setViewport:v]; s->viewport = v; s->valid |= 256; s->calls++;
    }
}
static inline void render_state_scissor(RenderState *s, id<MTLRenderCommandEncoder> e, MTLScissorRect v) {
    s->requests++;
    if (s->legacy || !(s->valid & 512) || memcmp(&s->scissor, &v, sizeof v)) {
        [e setScissorRect:v]; s->scissor = v; s->valid |= 512; s->calls++;
    }
}
