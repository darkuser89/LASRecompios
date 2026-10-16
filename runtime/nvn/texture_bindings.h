// Encoder-local state only: the encoder retains bound resources, so this cache must not retain them again.
#pragma once
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    __weak id<MTLRenderCommandEncoder> encoder;
    __unsafe_unretained id<MTLTexture> textures[2][64];
    __unsafe_unretained id<MTLSamplerState> samplers[2][64];
    uint64_t texture_valid[2], sampler_valid[2];
    uint64_t requests, calls;
} TextureBindings;

static inline void texture_bindings_reset(TextureBindings *b) {
    b->encoder = nil;
    b->texture_valid[0] = b->texture_valid[1] = 0;
    b->sampler_valid[0] = b->sampler_valid[1] = 0;
}

// Call once before binding a shader's slots. Pipeline changes preserve state, encoder changes do not.
static inline void texture_bindings_begin(TextureBindings *b, id<MTLRenderCommandEncoder> encoder) {
    if (b->encoder != encoder) {
        texture_bindings_reset(b);
        b->encoder = encoder;
    }
}

static inline void texture_bindings_apply(TextureBindings *b, id<MTLRenderCommandEncoder> encoder,
                                        unsigned stage, unsigned slot, id<MTLTexture> texture,
                                        id<MTLSamplerState> sampler, bool enabled, bool bind_sampler) {
    uint64_t mask = 1ull << slot;
    b->requests += bind_sampler ? 2 : 1;
    if (!enabled || !(b->texture_valid[stage] & mask) || b->textures[stage][slot] != texture) {
        if (stage == 0) [encoder setVertexTexture:texture atIndex:slot];
        else [encoder setFragmentTexture:texture atIndex:slot];
        b->textures[stage][slot] = texture;
        b->texture_valid[stage] |= mask;
        b->calls++;
    }
    if (bind_sampler && (!enabled || !(b->sampler_valid[stage] & mask) || b->samplers[stage][slot] != sampler)) {
        if (stage == 0) [encoder setVertexSamplerState:sampler atIndex:slot];
        else [encoder setFragmentSamplerState:sampler atIndex:slot];
        b->samplers[stage][slot] = sampler;
        b->sampler_valid[stage] |= mask;
        b->calls++;
    }
}

static inline void texture_bindings_set(TextureBindings *b, id<MTLRenderCommandEncoder> encoder,
                                        unsigned stage, unsigned slot, id<MTLTexture> texture,
                                        id<MTLSamplerState> sampler, bool enabled) {
    texture_bindings_apply(b, encoder, stage, slot, texture, sampler, enabled, true);
}
