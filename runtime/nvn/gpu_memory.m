#ifdef LARECOMP_COMPACT_GUEST
#import <Foundation/Foundation.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdatomic.h>
#include <time.h>
#include "gpu_memory.h"

typedef struct { uint64_t base, end; } PoolRange;
typedef struct { PoolRange range; unsigned refs; } PoolEntry;
static PoolEntry *pools;
static unsigned *pool_slots, slot_mask;
static unsigned pool_count, pool_capacity;
static _Atomic uint64_t generation;
static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;
static id<MTLDevice> gpu;
static id<MTLArgumentEncoder> argument_encoder;
static id<MTLBuffer> argument_buffer;
static NSMutableArray<id<MTLBuffer>> *buffers;
static PoolRange mapped[GPU_MEMORY_RANGES];
static unsigned mapped_count;
static uint64_t mapped_generation;
static id<MTLResource> __unsafe_unretained mapped_resources[GPU_MEMORY_RANGES];
static NSCache<NSString *, id<MTLBuffer>> *views;
static __weak id<MTLRenderCommandEncoder> resident_render;
static __weak id<MTLComputeCommandEncoder> resident_compute;
static __weak id<MTLBuffer> render_table, compute_table;
static MTLRenderStages resident_stages;
static MTLRenderStages resident_writers;
static bool compute_writes;
static uint64_t stat_binds, stat_residency_calls, stat_resources, stat_builds, stat_unchanged, stat_reused;
static uint64_t stat_bind_ns;

static bool diagnostic_stats(void) {
    static int on = -1;
    if (on < 0) on = getenv("LARECOMP_GPU_STATS") != NULL;
    return on;
}
static bool legacy_binding(void) {
    static int on = -1;
    if (on < 0) on = getenv("LARECOMP_GPU_BIND_LEGACY") != NULL;
    return on;
}
static bool conservative_usage(void) {
    static int on = -1;
    // Read-only residency is an opt-in profiling experiment: keep the measured stable path by default.
    if (on < 0) on = getenv("LARECOMP_GPU_RESIDENCY_READ_ONLY") == NULL || getenv("LARECOMP_GPU_RESIDENCY_RW") != NULL;
    return on;
}
void gpu_memory_log_stats(void) {
    if (!diagnostic_stats()) return;
    fprintf(stderr, "[gmem] binds %llu, residency calls %llu, resource refs %llu, tables %llu, unchanged %llu, reused buffers %llu, CPU %.3f ms (%s)\n",
            (unsigned long long)stat_binds, (unsigned long long)stat_residency_calls,
            (unsigned long long)stat_resources, (unsigned long long)stat_builds,
            (unsigned long long)stat_unchanged, (unsigned long long)stat_reused,
            stat_bind_ns / 1e6, legacy_binding() ? "legacy" : "batched");
    stat_binds = stat_residency_calls = stat_resources = stat_builds = stat_unchanged = stat_reused = stat_bind_ns = 0;
}

static void memory_fail(const char *message) { fprintf(stderr, "FATAL: GPU memory: %s\n", message); abort(); }
static unsigned range_hash(PoolRange r) {
    uint64_t h = r.base ^ (r.end + 0x9e3779b97f4a7c15ull + (r.base << 6) + (r.base >> 2));
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ull; h ^= h >> 27; h *= 0x94d049bb133111ebull;
    return (unsigned)(h ^ (h >> 31));
}
static unsigned range_slot(PoolRange r) {
    unsigned s = range_hash(r) & slot_mask;
    while (pool_slots[s]) {
        PoolRange p = pools[pool_slots[s] - 1].range;
        if (p.base == r.base && p.end == r.end) break;
        s = (s + 1) & slot_mask;
    }
    return s;
}
static void grow_registry(void) {
    unsigned capacity = pool_capacity ? pool_capacity * 2 : 256;
    if (capacity < pool_capacity || capacity > UINT_MAX / 2) memory_fail("buffer registry overflow");
    PoolEntry *next = realloc(pools, (size_t)capacity * sizeof *next);
    if (!next) memory_fail("cannot grow buffer registry");
    pools = next;
    unsigned *slots = calloc((size_t)capacity * 2, sizeof *slots);
    if (!slots) memory_fail("cannot grow buffer index");
    free(pool_slots); pool_slots = slots; slot_mask = capacity * 2 - 1; pool_capacity = capacity;
    for (unsigned i = 0; i < pool_count; i++) pool_slots[range_slot(pools[i].range)] = i + 1;
}
void gpu_pool_add(uint64_t address, uint64_t size) {
    if (!size) return;
    if (address + size < address) memory_fail("pool address overflow");
    pthread_mutex_lock(&pool_lock);
    PoolRange r = {address, address + size};
    if (!pool_capacity) grow_registry();
    unsigned s = range_slot(r);
    if (pool_slots[s]) {
        PoolEntry *p = &pools[pool_slots[s] - 1];
        if (p->refs == UINT_MAX) memory_fail("buffer reference overflow");
        p->refs++;
        pthread_mutex_unlock(&pool_lock);
        return;
    }
    if (pool_count == pool_capacity) { grow_registry(); s = range_slot(r); }
    pools[pool_count] = (PoolEntry){r, 1}; pool_slots[s] = ++pool_count;
    atomic_fetch_add_explicit(&generation, 1, memory_order_release);
    pthread_mutex_unlock(&pool_lock);
}
void gpu_pool_remove(uint64_t address, uint64_t size) {
    pthread_mutex_lock(&pool_lock);
    if (pool_count) {
        unsigned s = range_slot((PoolRange){address, address + size});
        if (pool_slots[s]) {
            unsigned i = pool_slots[s] - 1;
            if (--pools[i].refs == 0) {
                // Backshift the collision cluster, so deletion leaves no tombstones or unbounded probes.
                unsigned hole = s;
                for (unsigned k = (s + 1) & slot_mask; pool_slots[k]; k = (k + 1) & slot_mask) {
                    unsigned home = range_hash(pools[pool_slots[k] - 1].range) & slot_mask;
                    if (((k - home) & slot_mask) >= ((k - hole) & slot_mask)) {
                        pool_slots[hole] = pool_slots[k]; hole = k;
                    }
                }
                pool_slots[hole] = 0;
                unsigned last = --pool_count;
                if (i != last) {
                    pools[i] = pools[last];
                    pool_slots[range_slot(pools[i].range)] = i + 1;
                }
                atomic_fetch_add_explicit(&generation, 1, memory_order_release);
            }
        }
    }
    pthread_mutex_unlock(&pool_lock);
}
void gpu_memory_init(id<MTLDevice> device) {
    gpu = device;
    if (gpu.argumentBuffersSupport != MTLArgumentBuffersTier2) memory_fail("tier 2 argument buffers required");
    if (!argument_encoder) {
        MTLArgumentDescriptor *pointer = [MTLArgumentDescriptor argumentDescriptor];
        pointer.dataType = MTLDataTypePointer; pointer.arrayLength = GPU_MEMORY_RANGES; pointer.index = 0;
        pointer.access = MTLBindingAccessReadWrite;
        MTLArgumentDescriptor *ranges = [MTLArgumentDescriptor argumentDescriptor];
        ranges.dataType = MTLDataTypeULong2; ranges.arrayLength = GPU_MEMORY_RANGES; ranges.index = GPU_MEMORY_RANGES;
        MTLArgumentDescriptor *count = [MTLArgumentDescriptor argumentDescriptor];
        count.dataType = MTLDataTypeUInt; count.index = GPU_MEMORY_RANGES * 2;
        argument_encoder = [gpu newArgumentEncoderWithArguments:@[pointer, ranges, count]];
        if (!argument_encoder) memory_fail("cannot create pool table encoder");
    }
    if (!views) { views = [NSCache new]; views.countLimit = 256; }
}
void gpu_memory_reset(void) {
    resident_render = nil; resident_compute = nil; render_table = nil; compute_table = nil;
    resident_stages = resident_writers = 0; compute_writes = false;
    argument_buffer = nil; buffers = nil; mapped_count = 0; mapped_generation = 0;
    [views removeAllObjects];
    pthread_mutex_lock(&pool_lock);
    free(pools); pools = NULL; free(pool_slots); pool_slots = NULL;
    pool_count = pool_capacity = slot_mask = 0;
    atomic_fetch_add_explicit(&generation, 1, memory_order_release);
    pthread_mutex_unlock(&pool_lock);
}
static int compare_range(const void *a, const void *b) {
    uint64_t x = ((const PoolRange *)a)->base, y = ((const PoolRange *)b)->base;
    return x < y ? -1 : x > y;
}
static void update_pools(void) {
    if (argument_buffer && atomic_load_explicit(&generation, memory_order_acquire) == mapped_generation) return;
    PoolRange *active; unsigned n; uint64_t version;
    pthread_mutex_lock(&pool_lock);
    version = atomic_load_explicit(&generation, memory_order_relaxed);
    if (argument_buffer && version == mapped_generation) { pthread_mutex_unlock(&pool_lock); return; }
    n = pool_count;
    active = malloc((size_t)(n ? n : 1) * sizeof *active);
    if (!active) memory_fail("cannot snapshot buffer registry");
    for (unsigned i = 0; i < n; i++) active[i] = pools[i].range;
    pthread_mutex_unlock(&pool_lock);
    qsort(active, n, sizeof *active, compare_range);
    // Merge overlapping/adjacent pools, page-align for Metal, then split large spans into bounded buffers.
    unsigned merged = 0;
    for (unsigned i = 0; i < n; i++) {
        PoolRange r = {active[i].base & ~0x3fffull, (active[i].end + 0x3fff) & ~0x3fffull};
        if (merged && r.base <= active[merged - 1].end) {
            if (r.end > active[merged - 1].end) active[merged - 1].end = r.end;
        } else active[merged++] = r;
    }
    PoolRange ranges_next[GPU_MEMORY_RANGES];
    unsigned count = 0; uint64_t bytes = 0;
    for (unsigned i = 0; i < merged; i++) for (uint64_t a = active[i].base; a < active[i].end;) {
        uint64_t length = active[i].end - a;
        if (length > (16ull << 20)) length = 16ull << 20;
        if (count == GPU_MEMORY_RANGES) memory_fail("too many pool segments");
        ranges_next[count++] = (PoolRange){a, a + length}; bytes += length; a += length;
    }
    free(active);
    // Thousands of small buffer objects can occupy pages already covered by the current table.
    if (argument_buffer && count == mapped_count && !memcmp(ranges_next, mapped, count * sizeof *mapped)) {
        mapped_generation = version; stat_unchanged++; return;
    }
    NSMutableArray<id<MTLBuffer>> *next = [NSMutableArray new];
    id<MTLBuffer> table = [gpu newBufferWithLength:argument_encoder.encodedLength options:MTLResourceStorageModeShared];
    if (!table) memory_fail("cannot allocate pool table");
    memset(table.contents, 0, table.length);
    [argument_encoder setArgumentBuffer:table offset:0];
    uint64_t *ranges = [argument_encoder constantDataAtIndex:GPU_MEMORY_RANGES];
    for (unsigned i = 0; i < count; i++) {
        uint64_t a = ranges_next[i].base, length = ranges_next[i].end - a;
        unsigned lo = 0, hi = mapped_count;
        while (lo < hi) { unsigned mid = (lo + hi) / 2; if (mapped[mid].base < a) lo = mid + 1; else hi = mid; }
        id<MTLBuffer> buffer = nil;
        if (lo < mapped_count && mapped[lo].base == a && mapped[lo].end == a + length) {
            buffer = buffers[lo]; stat_reused++;
        } else buffer = [gpu newBufferWithBytesNoCopy:(void *)(uintptr_t)a length:length
                                             options:MTLResourceStorageModeShared deallocator:nil];
        if (!buffer) memory_fail("cannot map pool segment");
        [next addObject:buffer];
        [argument_encoder setBuffer:buffer offset:0 atIndex:i];
        ranges[i * 2] = a; ranges[i * 2 + 1] = length;
    }
    *(uint32_t *)[argument_encoder constantDataAtIndex:GPU_MEMORY_RANGES * 2] = count;
    // Tables are immutable once submitted. Old command buffers retain both the old table and its resources.
    argument_buffer = table; buffers = next; mapped_count = count; mapped_generation = version;
    memcpy(mapped, ranges_next, count * sizeof *mapped);
    for (unsigned i = 0; i < count; i++) mapped_resources[i] = buffers[i];
    stat_builds++;
    [views removeAllObjects];
    fprintf(stderr, "[metal] buffer mappings: %u buffers, %u segments, %llu MB pinned (CPU heap remains pageable)\n",
            n, count, (unsigned long long)(bytes >> 20));
}
id<MTLBuffer> gpu_guest_buffer(uint64_t address, uint64_t size, uint64_t *offset) {
    if (!size || address + size < address) memory_fail("invalid buffer range");
    update_pools();
    for (unsigned i = 0; i < mapped_count; i++) if (address >= mapped[i].base && address + size <= mapped[i].end) {
        *offset = address - mapped[i].base; return buffers[i];
    }
    uint64_t lo = address & ~0x3fffull, hi = (address + size + 0x3fff) & ~0x3fffull;
    NSString *key = [NSString stringWithFormat:@"%llx/%llx", lo, hi];
    id<MTLBuffer> buffer = [views objectForKey:key];
    if (!buffer) {
        buffer = [gpu newBufferWithBytesNoCopy:(void *)(uintptr_t)lo length:hi - lo
                                     options:MTLResourceStorageModeShared deallocator:nil];
        if (!buffer) memory_fail("cannot map buffer view");
        [views setObject:buffer forKey:key];
    }
    *offset = address - lo; return buffer;
}
void gpu_bind_render(id<MTLRenderCommandEncoder> encoder, bool vertex, bool writes) {
    uint64_t start = diagnostic_stats() ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;
    stat_binds++;
    update_pools();
    MTLRenderStages stage = vertex ? MTLRenderStageVertex : MTLRenderStageFragment;
    writes |= conservative_usage();
    MTLResourceUsage usage = MTLResourceUsageRead | (writes ? MTLResourceUsageWrite : 0);
    if (legacy_binding()) {
        for (id<MTLBuffer> buffer in buffers)
            [encoder useResource:buffer usage:usage stages:stage];
        stat_residency_calls += mapped_count; stat_resources += mapped_count;
    } else {
        if (resident_render != encoder || render_table != argument_buffer) {
            resident_render = encoder; render_table = argument_buffer; resident_stages = resident_writers = 0;
        }
        if (!(resident_stages & stage) || (writes && !(resident_writers & stage))) {
            if (mapped_count) {
                [encoder useResources:mapped_resources count:mapped_count
                               usage:usage stages:stage];
                stat_residency_calls++; stat_resources += mapped_count;
            }
            resident_stages |= stage;
            if (writes) resident_writers |= stage;
        }
    }
    // Constant-buffer bindings can overwrite slot 17 between draws: always restore the table itself.
    if (vertex) [encoder setVertexBuffer:argument_buffer offset:0 atIndex:17];
    else [encoder setFragmentBuffer:argument_buffer offset:0 atIndex:17];
    if (start) stat_bind_ns += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start;
}
void gpu_bind_compute(id<MTLComputeCommandEncoder> encoder, bool writes) {
    uint64_t start = diagnostic_stats() ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;
    stat_binds++;
    update_pools();
    writes |= conservative_usage();
    MTLResourceUsage usage = MTLResourceUsageRead | (writes ? MTLResourceUsageWrite : 0);
    if (legacy_binding()) {
        for (id<MTLBuffer> buffer in buffers) [encoder useResource:buffer usage:usage];
        stat_residency_calls += mapped_count; stat_resources += mapped_count;
    } else if (resident_compute != encoder || compute_table != argument_buffer || (writes && !compute_writes)) {
        resident_compute = encoder; compute_table = argument_buffer; compute_writes = writes;
        if (mapped_count) {
            [encoder useResources:mapped_resources count:mapped_count usage:usage];
            stat_residency_calls++; stat_resources += mapped_count;
        }
    }
    [encoder setBuffer:argument_buffer offset:0 atIndex:17];
    if (start) stat_bind_ns += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start;
}
#endif
