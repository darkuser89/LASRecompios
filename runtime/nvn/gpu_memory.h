// iOS: expose active NVN buffer ranges to Metal, rather than wiring entire pools or the CPU heap.
#pragma once
#include <stdint.h>
#include <stdbool.h>
#define GPU_MEMORY_RANGES 2048
#define GPU_MEMORY_MSL \
"struct GPUMemory { array<device uchar *, 2048> pages [[id(0)]]; array<ulong2, 2048> ranges [[id(2048)]]; uint count [[id(4096)]]; };\n" \
"static inline device uchar *gpu_guest_ptr(constant GPUMemory &m, ulong a, uint bytes = 1u) {\n" \
"  uint lo = 0, hi = m.count; while (lo < hi) { uint i = (lo + hi) / 2; ulong2 r = m.ranges[i];\n" \
"    if (a < r.x) hi = i; else if (a - r.x >= r.y) lo = i + 1; else return bytes <= r.y - (a - r.x) ? m.pages[i] + (a - r.x) : nullptr; } return nullptr; }\n" \
"static inline uint gpu_load32(constant GPUMemory &m, ulong a) { device uchar *p = gpu_guest_ptr(m, a, 4u); return p ? *(device uint *)p : 0u; }\n" \
"static inline void gpu_store32(constant GPUMemory &m, ulong a, uint v) { device uchar *p = gpu_guest_ptr(m, a, 4u); if (p) *(device uint *)p = v; }\n"

void gpu_pool_add(uint64_t address, uint64_t size);
void gpu_pool_remove(uint64_t address, uint64_t size);
#ifdef __OBJC__
#import <Metal/Metal.h>
void gpu_memory_init(id<MTLDevice> device);
void gpu_memory_reset(void);  // GPU idle; old commands have released their resources
void gpu_memory_log_stats(void);  // optional LARECOMP_GPU_STATS counters, reset after logging
id<MTLBuffer> gpu_guest_buffer(uint64_t address, uint64_t size, uint64_t *offset);
void gpu_bind_render(id<MTLRenderCommandEncoder> encoder, bool vertex, bool writes);
void gpu_bind_compute(id<MTLComputeCommandEncoder> encoder, bool writes);
#endif
