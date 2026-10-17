// Share translation by stage and compilation by exact source/options. Per-entry locks coalesce
// concurrent requests without serializing unrelated shader compiles. Entries own their metadata.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "maxwell/shader.h"
#include <stdlib.h>
#include <string.h>

@interface MetalShaderPart : NSObject {
@public
    ShaderInfo *info;
}
@end
@implementation MetalShaderPart
- (void)dealloc { if (info) maxwell_free(info); }
@end

@interface MetalShaderFunction : NSObject {
@public
    id<MTLFunction> function;
    bool attempted;
    NSString *error;
}
@end
@implementation MetalShaderFunction
@end

typedef struct {
    NSMutableDictionary<NSData *, MetalShaderPart *> *parts;
    NSCache<NSString *, MetalShaderFunction *> *functions;
    uint64_t translations, translation_hits, compilations, compilation_hits;
} MetalShaderCache;

static inline void metal_shader_cache_reset(MetalShaderCache *c) {
    @synchronized([MetalShaderPart class]) { c->parts = nil; }
    // Exact source/options are independent of guest addresses. Keep the bounded function cache
    // through a reboot, including entries whose compilation is still in flight.
}

static inline MetalShaderPart *metal_shader_part_source(MetalShaderCache *c, uint64_t address, uint64_t code_hash,
                                                size_t length, int stage, const ShaderKey *key, const uint8_t *source) {
    // The translator still has shared scratch buffers. Keep translation/coalescing serialized,
    // while the much slower Metal function and pipeline compiles can run on both workers.
    @synchronized([MetalShaderPart class]) {
    struct {
        uint64_t address, code_hash, length, debug_code, debug_vs, debug_hash;
        uint32_t debug_pc, debug_reg, stage;
        ShaderKey key;
    } identity;
    memset(&identity, 0, sizeof identity);
    identity.address = address; identity.code_hash = code_hash; identity.length = length; identity.stage = stage;
    identity.debug_code = (uintptr_t)maxwell_dbg_code; identity.debug_vs = (uintptr_t)maxwell_dbg_vs;
    identity.debug_hash = maxwell_dbg_hash; identity.debug_pc = maxwell_dbg_pc; identity.debug_reg = maxwell_dbg_reg;
    // Only inputs consumed by this stage belong in its translation key.
    if (stage == STAGE_VERTEX) {
        memcpy(identity.key.attr_kind, key->attr_kind, sizeof key->attr_kind);
        identity.key.depth_minus_one = key->depth_minus_one;
        identity.key.flip_y = key->flip_y;
    } else {
        memcpy(identity.key.rt_kind, key->rt_kind, sizeof key->rt_kind);
        identity.key.vs_out_mask = key->vs_out_mask;
    }
    identity.key.flip_x = key->flip_x; identity.key.glsl = key->glsl;
    NSData *k = [NSData dataWithBytes:&identity length:sizeof identity];
    if (!c->parts) c->parts = [NSMutableDictionary new];
    MetalShaderPart *part = c->parts[k];
    if (part) { __atomic_fetch_add(&c->translation_hits, 1, __ATOMIC_RELAXED); return part; }
    part = [MetalShaderPart new];
    part->info = maxwell_translate_snapshot(source, length, stage, &identity.key, address);
    c->parts[k] = part; __atomic_fetch_add(&c->translations, 1, __ATOMIC_RELAXED);
    return part;
    }
}

static inline MetalShaderPart *metal_shader_part(MetalShaderCache *c, uint64_t address, uint64_t code_hash,
                                                size_t length, int stage, const ShaderKey *key) {
    return metal_shader_part_source(c, address, code_hash, length, stage, key, (const uint8_t *)(uintptr_t)address);
}

static inline id<MTLFunction> metal_shader_function(MetalShaderCache *c, id<MTLDevice> dev, const char *msl,
                                                   bool vertex, NSString **error) {
    static NSString *salt;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        const char *e = getenv("LARECOMP_SHADER_SALT");
        salt = e ? [NSString stringWithFormat:@"\n// %s\n", e] : @"";
    });
    NSString *source = [@(msl) stringByAppendingString:salt];
    NSString *key = [NSString stringWithFormat:@"%c%@", vertex ? 'v' : 'f', source];
    @synchronized(dev) {
        if (!c->functions) { c->functions = [NSCache new]; c->functions.countLimit = 1024; }
    }
    // Keep a local reference while unrelated shader jobs access the shared cache.
    NSCache *functions = c->functions;
    MetalShaderFunction *entry;
    @synchronized(functions) {
        entry = [functions objectForKey:key];
        if (!entry) { entry = [MetalShaderFunction new]; [functions setObject:entry forKey:key]; }
    }
    @synchronized(entry) {
        if (!entry->attempted) {
            entry->attempted = true;
            MTLCompileOptions *options = nil;
            if (vertex) { options = [MTLCompileOptions new]; options.preserveInvariance = YES; }
            NSError *e = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:source options:options error:&e];
            entry->function = [lib newFunctionWithName:@"main0"];
            entry->error = e.localizedDescription;
            __atomic_fetch_add(&c->compilations, 1, __ATOMIC_RELAXED);
        } else __atomic_fetch_add(&c->compilation_hits, 1, __ATOMIC_RELAXED);
        if (error) *error = entry->error;
        return entry->function;
    }
}
