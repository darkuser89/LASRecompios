// Immutable compiler constants are shared between draws/stages. Compare source bytes before reuse:
// guest addresses may be recycled or rewritten; never cache mutable uniform snapshots this way.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    NSCache<NSData *, id<MTLBuffer>> *cache;
    id<MTLBuffer> hot[4];
    uint64_t addresses[4], sizes[4];
    unsigned next;
    uint64_t requests, copies, bytes;
} ShaderBuffers;

static inline void shader_buffers_reset(ShaderBuffers *s) {
    s->cache = nil;
    for (unsigned i = 0; i < 4; i++) { s->hot[i] = nil; s->sizes[i] = 0; }
    s->next = 0;
}

static inline id<MTLBuffer> shader_constant_buffer(ShaderBuffers *s, id<MTLDevice> dev, uint64_t addr, uint64_t size) {
    s->requests++;
    const void *data = (const void *)(uintptr_t)addr;
    for (unsigned i = 0; i < 4; i++) {
        if (s->hot[i] && s->addresses[i] == addr && s->sizes[i] == size && !memcmp(data, s->hot[i].contents, size))
            return s->hot[i];
    }
    uint64_t words[2] = {addr, size};
    NSData *key = [NSData dataWithBytes:words length:sizeof words];
    if (!s->cache) { s->cache = [NSCache new]; s->cache.countLimit = 256; s->cache.totalCostLimit = 8 << 20; }
    id<MTLBuffer> buffer = [s->cache objectForKey:key];
    if (!buffer || memcmp(data, buffer.contents, size)) {
        buffer = [dev newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
        if (!buffer) return nil;
        [s->cache setObject:buffer forKey:key cost:size];
        s->copies++; s->bytes += size;
    }
    unsigned i = s->next++ & 3;
    s->hot[i] = buffer; s->addresses[i] = addr; s->sizes[i] = size;
    return buffer;
}
