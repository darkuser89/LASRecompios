// Metal permits 16 direct sampler arguments. Larger shaders use an argument buffer, retaining
// its sampler objects with the buffer so command buffers in flight never reference dead handles.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    id<MTLArgumentEncoder> encoder;
    NSCache<NSData *, id<MTLBuffer>> *cache;
    // Retained tables retain their samplers; pointer comparisons here cannot hit reused objects.
    id<MTLBuffer> hot[4];
    uintptr_t hot_samplers[4][32];
    unsigned hot_count[4], next_hot;
    uint64_t hot_hits, key_builds;
} SamplerTables;

static inline void sampler_tables_reset(SamplerTables *tables) {
    tables->encoder = nil;
    tables->cache = nil;
    for (unsigned i = 0; i < 4; i++) { tables->hot[i] = nil; tables->hot_count[i] = 0; }
    tables->next_hot = 0;
}

static inline id<MTLBuffer> sampler_table_remember(SamplerTables *tables, id<MTLBuffer> buffer,
                                                   const uintptr_t *addresses, unsigned count) {
    unsigned hot = tables->next_hot++ & 3;
    tables->hot[hot] = buffer;
    tables->hot_count[hot] = count;
    memcpy(tables->hot_samplers[hot], addresses, count * sizeof addresses[0]);
    return buffer;
}

static inline id<MTLBuffer> sampler_table_buffer(SamplerTables *tables, id<MTLDevice> device,
                                                id<MTLSamplerState> const __unsafe_unretained *samplers, unsigned count) {
    assert(count > 16 && count <= 32);
    for (unsigned k = 0; k < 4; k++) {
        if (!tables->hot[k] || tables->hot_count[k] != count) continue;
        unsigned i = 0;
        while (i < count && tables->hot_samplers[k][i] == (uintptr_t)(__bridge void *)samplers[i]) i++;
        if (i == count) { tables->hot_hits++; return tables->hot[k]; }
    }
    uintptr_t addresses[32];
    for (unsigned i = 0; i < count; i++) addresses[i] = (uintptr_t)(__bridge void *)samplers[i];
    NSData *key = [NSData dataWithBytes:addresses length:count * sizeof addresses[0]];
    tables->key_builds++;
    id<MTLBuffer> buffer = [tables->cache objectForKey:key];
    if (buffer) return sampler_table_remember(tables, buffer, addresses, count);
    if (!tables->encoder) {
        MTLArgumentDescriptor *d = [MTLArgumentDescriptor argumentDescriptor];
        d.dataType = MTLDataTypeSampler; d.index = 0; d.arrayLength = 32;
        tables->encoder = [device newArgumentEncoderWithArguments:@[d]];
        tables->cache = [NSCache new]; tables->cache.countLimit = 64;
    }
    if (!tables->encoder) return nil;
    buffer = [device newBufferWithLength:tables->encoder.encodedLength options:MTLResourceStorageModeShared];
    if (!buffer) return nil;
    [tables->encoder setArgumentBuffer:buffer offset:0];
    NSMutableArray *keep = [NSMutableArray arrayWithCapacity:count];
    for (unsigned i = 0; i < 32; i++) {
        [tables->encoder setSamplerState:i < count ? samplers[i] : nil atIndex:i];
        if (i < count) [keep addObject:samplers[i]];
    }
    static char lifetime_key;
    objc_setAssociatedObject((id)buffer, &lifetime_key, [keep copy], OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [tables->cache setObject:buffer forKey:key];
    return sampler_table_remember(tables, buffer, addresses, count);
}
