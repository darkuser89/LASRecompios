// Buffer bindings are encoder-local. Uniform snapshots still get distinct offsets; only an identical
// buffer/offset pair is suppressed. Inline bytes and external bindings invalidate their slot.
#pragma once
#import <Metal/Metal.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    __unsafe_unretained id<MTLBuffer> buffers[2][31];
    NSUInteger offsets[2][31];
    uint32_t valid[2];
    uint64_t requests, binds, offsets_changed, skipped;
    bool legacy;
} BufferBindings;

static inline void buffer_bindings_reset(BufferBindings *b) { b->valid[0] = b->valid[1] = 0; }
static inline void buffer_bindings_invalidate(BufferBindings *b, unsigned stage, unsigned slot) {
    b->valid[stage] &= ~(1u << slot);
}

static inline void buffer_bindings_set(BufferBindings *b, id<MTLRenderCommandEncoder> enc, unsigned stage,
                                       unsigned slot, id<MTLBuffer> buffer, NSUInteger offset) {
    uint32_t bit = 1u << slot;
    b->requests++;
    bool same = !b->legacy && (b->valid[stage] & bit) && b->buffers[stage][slot] == buffer;
    if (same && b->offsets[stage][slot] == offset) { b->skipped++; return; }
    if (same && buffer) {
        if (stage == 0) [enc setVertexBufferOffset:offset atIndex:slot];
        else [enc setFragmentBufferOffset:offset atIndex:slot];
        b->offsets_changed++;
    } else {
        if (stage == 0) [enc setVertexBuffer:buffer offset:offset atIndex:slot];
        else [enc setFragmentBuffer:buffer offset:offset atIndex:slot];
        b->binds++;
    }
    b->buffers[stage][slot] = buffer;
    b->offsets[stage][slot] = offset;
    b->valid[stage] |= bit;
}

static inline void buffer_bindings_bytes(BufferBindings *b, id<MTLRenderCommandEncoder> enc, unsigned stage,
                                         unsigned slot, const void *bytes, NSUInteger size) {
    buffer_bindings_invalidate(b, stage, slot);
    if (stage == 0) [enc setVertexBytes:bytes length:size atIndex:slot];
    else [enc setFragmentBytes:bytes length:size atIndex:slot];
}
