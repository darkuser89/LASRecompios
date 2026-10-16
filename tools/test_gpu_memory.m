// Real Metal test: discontiguous pools, unmapped gaps, segment boundaries and GPU writes back to guest RAM.
#import <Foundation/Foundation.h>
#include <assert.h>
#include <mach/mach.h>
#include "../runtime/nvn/gpu_memory.h"
#include "../runtime/hle/resource_cache.h"

static const char *source = "#include <metal_stdlib>\nusing namespace metal;\n" GPU_MEMORY_MSL
"kernel void check(constant GPUMemory &m [[buffer(17)]], device uint *out [[buffer(0)]],\n"
" constant ulong *addresses [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
" ulong a = addresses[i]; uint v = gpu_load32(m, a); out[i] = v; gpu_store32(m, a, v ^ 0x00ff00ffu); }\n"
"kernel void check_span(constant GPUMemory &m [[buffer(17)]], device uint *out [[buffer(0)]],\n"
" constant ulong &a [[buffer(1)]]) { device uchar *p = gpu_guest_ptr(m,a,16u);\n"
" for (uint k=0;k<4;k++) { uint v = p ? *(device uint *)(p+k*4) : gpu_load32(m,a+k*4); out[k]=v;\n"
" if (p) *(device uint *)(p+k*4)=v^0x01010101u; else gpu_store32(m,a+k*4,v^0x01010101u); } }\n"
"struct Varying { float4 position [[position]]; uint value [[flat]]; };\n"
"vertex Varying read_vertex(uint i [[vertex_id]], constant GPUMemory &m [[buffer(17)]],\n"
" constant ulong *a [[buffer(0)]]) { const float2 p[] = {float2(-1,-1),float2(3,-1),float2(-1,3)};\n"
" return {float4(p[i],0,1),gpu_load32(m,a[0])}; }\n"
"fragment uint read_fragment(Varying v [[stage_in]], constant GPUMemory &m [[buffer(17)]],\n"
" constant ulong *a [[buffer(0)]]) { return v.value ^ gpu_load32(m,a[1]); }\n";

int main(void) { @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    assert(device);
    const vm_size_t total = 32ull << 20;
    const size_t guard_size = 0x4000;
    uint8_t *reservation = NULL;
    void *cache;
    if (getenv("LARECOMP_TEST_FIXED_CACHE")) {
        reservation = mmap(NULL, total + 2 * guard_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        assert(reservation != MAP_FAILED);
        memset(reservation, 0x5a, guard_size);
        memset(reservation + guard_size + total, 0xa5, guard_size);
        cache = resource_cache_map_at(reservation + guard_size, total);
        assert(cache == reservation + guard_size);
    } else cache = resource_cache_map(total);
    assert(cache);
    vm_address_t allocation = (uintptr_t)cache;
    uint64_t a = allocation + 0x4000, b = allocation + (4ull << 20), extra = allocation + (3ull << 20);
    const uint64_t asize = 0x10000, bsize = 20ull << 20;
    uint64_t addresses[] = {a, a + asize - 4, a + asize, b, b + (16ull << 20) - 8,
        b + (16ull << 20) - 4, b + (16ull << 20), b + (16ull << 20) + 4, b + bsize - 4, extra, b + bsize};
    const unsigned count = sizeof addresses / sizeof *addresses;
    for (unsigned i = 0; i < count; i++) *(uint32_t *)(uintptr_t)addresses[i] = 0xa0000000u + i;
    gpu_memory_init(device);
    gpu_pool_add(a, asize); gpu_pool_add(b, bsize);
    gpu_pool_add(b, bsize);  // shared storage: one owner leaving must preserve the other owner's mapping
    // The overworld has more than 4096 small NVN buffers. Registry size must not limit that scene;
    // overlapping page-aligned ranges still coalesce into a small GPU table.
    for (unsigned i = 0; i < 8192; i++) gpu_pool_add(a + i * 4, 4);
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(source) options:nil error:&error];
    if (!library) fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
    assert(library);
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"check"] error:&error];
    assert(pipeline);
    id<MTLComputePipelineState> span_pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"check_span"] error:&error];
    assert(span_pipeline);
    id<MTLBuffer> output = [device newBufferWithLength:count * 4 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    for (int pass = 0; pass < 3; pass++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:output offset:0 atIndex:0];
        [encoder setBytes:addresses length:sizeof addresses atIndex:1];
        gpu_bind_compute(encoder, false);  // a later shader on the same encoder can upgrade read to read/write
        gpu_bind_compute(encoder, true);
        if (!pass) {
            gpu_pool_add(a + 0x100, 8);  // unchanged page coverage: reuse the current table
            gpu_bind_compute(encoder, true);
            gpu_pool_add(extra, 4);  // changed coverage on the SAME encoder: declare the new resources
        }
        gpu_bind_compute(encoder, true);  // repeated binding on one encoder must preserve residency
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(count, 1, 1)];
        [encoder endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) fprintf(stderr, "%s\n", cb.error.localizedDescription.UTF8String);
        assert(cb.status == MTLCommandBufferStatusCompleted);
        uint32_t *values = output.contents;
        for (unsigned i = 0; i < count; i++) {
            bool mapped = i != 2 && i != count - 1 && !(pass && i < 2) && pass < 2;
            uint32_t expected = mapped ? (0xa0000000u + i) ^ (pass ? 0x00ff00ffu : 0) : 0;
            assert(values[i] == expected);
            if (mapped) assert(*(uint32_t *)(uintptr_t)addresses[i] == (expected ^ 0x00ff00ffu));
        }
        if (!pass) for (unsigned i = 0; i < count; i++) {
            uint32_t expected[4], before[4]; bool valid[4];
            for (unsigned k = 0; k < 4; k++) {
                uint64_t v = addresses[i] + k * 4;
                valid[k] = (v >= a && v < a + asize) || (v >= b && v < b + bsize) ||
                           (v >= extra && v < extra + 0x4000);
                before[k] = *(uint32_t *)(uintptr_t)v;
                expected[k] = valid[k] ? before[k] : 0;
            }
            id<MTLCommandBuffer> span_cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> span_encoder = [span_cb computeCommandEncoder];
            [span_encoder setComputePipelineState:span_pipeline];
            [span_encoder setBuffer:output offset:0 atIndex:0];
            [span_encoder setBytes:&addresses[i] length:8 atIndex:1];
            gpu_bind_compute(span_encoder, true);
            [span_encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [span_encoder endEncoding]; [span_cb commit]; [span_cb waitUntilCompleted];
            assert(span_cb.status == MTLCommandBufferStatusCompleted);
            for (unsigned k = 0; k < 4; k++) {
                assert(((uint32_t *)output.contents)[k] == expected[k]);
                if (valid[k]) {
                    uint32_t *v = (uint32_t *)(uintptr_t)(addresses[i] + k * 4);
                    assert(*v == (expected[k] ^ 0x01010101u));
                    *v = expected[k];
                } else assert(*(uint32_t *)(uintptr_t)(addresses[i] + k * 4) == before[k]);
            }
        }
        gpu_pool_remove(a, asize);
        if (!pass) {
            for (unsigned i = 0; i < 8192; i++) gpu_pool_remove(a + i * 4, 4);
            gpu_pool_remove(a + 0x100, 8);
        }
        if (pass < 2) gpu_pool_remove(b, bsize);
        if (pass == 1) gpu_pool_remove(extra, 4);
    }
    uint64_t offset;
    id<MTLBuffer> view = gpu_guest_buffer(b + 0x100, 64, &offset);
    assert((uint8_t *)view.contents + offset == (uint8_t *)(uintptr_t)(b + 0x100));
    gpu_memory_reset();
    // Probe shuffled hash-table deletion and shared ownership through GPU-visible holes. One value per
    // page prevents page rounding from hiding a stale registry entry after its last owner is removed.
    enum { STRESS = 2048 };
    uint64_t stress_addresses[STRESS];
    unsigned order[STRESS], refs[STRESS];
    uint32_t random = 0x12345678;
    for (unsigned i = 0; i < STRESS; i++) {
        stress_addresses[i] = allocation + i * 0x4000ull + 0x100;
        *(uint32_t *)(uintptr_t)stress_addresses[i] = 0xb0000000u + i;
        order[i] = i; refs[i] = i % 7 == 0 ? 2 : 1;
        for (unsigned j = 0; j < refs[i]; j++) gpu_pool_add(stress_addresses[i], 4);
    }
    for (unsigned i = STRESS - 1; i; i--) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        unsigned j = random % (i + 1), swap = order[i]; order[i] = order[j]; order[j] = swap;
    }
    id<MTLBuffer> stress_input = [device newBufferWithBytes:stress_addresses length:sizeof stress_addresses
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> stress_output = [device newBufferWithLength:STRESS * 4 options:MTLResourceStorageModeShared];
    for (unsigned pass = 0; pass < 3; pass++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:stress_output offset:0 atIndex:0];
        [encoder setBuffer:stress_input offset:0 atIndex:1];
        gpu_bind_compute(encoder, false); gpu_bind_compute(encoder, true); gpu_bind_compute(encoder, true);
        [encoder dispatchThreads:MTLSizeMake(STRESS, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding]; [cb commit]; [cb waitUntilCompleted];
        assert(cb.status == MTLCommandBufferStatusCompleted);
        for (unsigned i = 0; i < STRESS; i++) {
            uint32_t expected = refs[i] ? (0xb0000000u + i) ^ (pass ? 0x00ff00ffu : 0) : 0;
            assert(((uint32_t *)stress_output.contents)[i] == expected);
        }
        for (unsigned j = 0; j < (pass ? STRESS : STRESS / 2); j++) {
            unsigned i = order[j];
            if (!pass) { gpu_pool_remove(stress_addresses[i], 4); refs[i]--; }
            else while (refs[i]) { gpu_pool_remove(stress_addresses[i], 4); refs[i]--; }
        }
    }
    gpu_memory_reset();
    // Render stages must both declare indirect resources. Rebinding must also restore slot 17 after
    // another shader's constant buffers overwrite it, even if residency was already declared.
    uint64_t render_addresses[] = {a + 0x100, b + 0x100};
    *(uint32_t *)(uintptr_t)render_addresses[0] = 0x12345678;
    *(uint32_t *)(uintptr_t)render_addresses[1] = 0xabcdef01;
    gpu_pool_add(render_addresses[0], 4);
    MTLRenderPipelineDescriptor *render_desc = [MTLRenderPipelineDescriptor new];
    render_desc.vertexFunction = [library newFunctionWithName:@"read_vertex"];
    render_desc.fragmentFunction = [library newFunctionWithName:@"read_fragment"];
    render_desc.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
    id<MTLRenderPipelineState> render_pipeline = [device newRenderPipelineStateWithDescriptor:render_desc error:&error];
    assert(render_pipeline);
    MTLTextureDescriptor *tex_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                                    width:1 height:1 mipmapped:NO];
    tex_desc.storageMode = MTLStorageModeShared; tex_desc.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> result = [device newTextureWithDescriptor:tex_desc];
    assert(result);
    for (unsigned pass = 0; pass < 2; pass++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor new];
        rp.colorAttachments[0].texture = result;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> encoder = [cb renderCommandEncoderWithDescriptor:rp];
        [encoder setRenderPipelineState:render_pipeline];
        [encoder setVertexBytes:render_addresses length:sizeof render_addresses atIndex:0];
        [encoder setFragmentBytes:render_addresses length:sizeof render_addresses atIndex:0];
        gpu_bind_render(encoder, true, false);
        if (!pass) gpu_pool_add(render_addresses[1], 4);  // table changes between vertex and fragment bindings
        gpu_bind_render(encoder, false, false);
        [encoder setVertexBuffer:output offset:0 atIndex:17];
        gpu_bind_render(encoder, true, false); gpu_bind_render(encoder, false, false);
        if (pass) { gpu_bind_render(encoder, true, true); gpu_bind_render(encoder, true, false); }
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding]; [cb commit]; [cb waitUntilCompleted];
        assert(cb.status == MTLCommandBufferStatusCompleted);
        uint32_t pixel = 0;
        [result getBytes:&pixel bytesPerRow:4 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
        assert(pixel == (0x12345678u ^ 0xabcdef01u));
    }
    gpu_memory_reset();
    gpu_memory_log_stats();
    // A file page can be clean before a shader writes it. After unwiring the NoCopy resource,
    // forced invalidation must reload the GPU's new bytes from disk, not an older clean version.
    void *clean_page = resource_cache_map(0x4000);
    assert(clean_page);
    uint64_t clean_address = (uintptr_t)clean_page;
    *(uint32_t *)clean_page = 0x13579bdfu;
    @autoreleasepool {
        gpu_pool_add(clean_address, 4);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:output offset:0 atIndex:0];
        [encoder setBytes:&clean_address length:sizeof clean_address atIndex:1];
        gpu_bind_compute(encoder, true);
        assert(msync(clean_page, 0x4000, MS_SYNC) == 0);
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding]; [cb commit]; [cb waitUntilCompleted];
        assert(cb.status == MTLCommandBufferStatusCompleted);
        assert(*(uint32_t *)clean_page == (0x13579bdfu ^ 0x00ff00ffu));
        gpu_memory_reset();
    }
    assert(msync(clean_page, 0x4000, MS_SYNC | MS_INVALIDATE) == 0);
    assert(*(uint32_t *)clean_page == (0x13579bdfu ^ 0x00ff00ffu));
    assert(munmap(clean_page, 0x4000) == 0);
    assert(msync(cache, total, MS_SYNC) == 0);
    assert(madvise(cache, total, MADV_DONTNEED) == 0);
    assert(*(uint32_t *)(uintptr_t)b == 0xa0000003u);
    assert(munmap(cache, total) == 0);
    if (reservation) {
        for (size_t i = 0; i < guard_size; i++) {
            assert(reservation[i] == 0x5a);
            assert(reservation[guard_size + total + i] == 0xa5);
        }
        // A reboot installs a new zeroed backing in the same owned reservation.
        assert(mmap(reservation + guard_size, total, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) == reservation + guard_size);
        cache = resource_cache_map_at(reservation + guard_size, total);
        assert(cache == reservation + guard_size);
        for (size_t i = 0; i < total; i += guard_size) assert(((uint8_t *)cache)[i] == 0);
        assert(munmap(reservation, total + 2 * guard_size) == 0);
    }
    puts("Metal GPU mapping: file-backed cache, 8192+ buffers, gaps, segment boundaries, writes, views, removal and writeback passed");
    return 0;
} }
