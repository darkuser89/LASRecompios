// Game heaps (engine limits). The game carves its memory into SDK heaps of fixed sizes: nn::mem::StandardAllocator
// (general), nn::lmem unit heaps (fixed-size blocks: object pools) and frame heaps. A heap that runs out returns
// NULL and the game crashes further on (bushsanity: ~450 actors around Link -> null weak_ptr in f_70010d0f00).
// These wrappers call the SDK's own implementation (loader_find_symbol), log heap creation (LARECOMP_LOG_HEAPS=1)
// and every allocation that fails (once per caller), and can grow heaps (heap_scale) - a recomp has memory to spare.
#include <pthread.h>
#ifdef _WIN32
#include <malloc.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../hle.h"
#ifdef LARECOMP_COMPACT_GUEST
#include <stdatomic.h>
#include "resource_cache.h"
#endif
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

uint64_t loader_find_symbol(const char *name);
void guest_call(Cpu *c, uint64_t addr);

static uint64_t orig(const char *name, uint64_t *cache) {
    if (!*cache) *cache = loader_find_symbol(name);
    if (!*cache) FATAL("heaps: no SDK implementation of %s\n", name);
    return *cache;
}

static void fail_once(const char *what, uint64_t lr, uint64_t size) {
    static uint64_t seen[64];
    static int n;
    for (int i = 0; i < n; i++) if (seen[i] == lr) return;
    if (n < 64) seen[n++] = lr;
    LOG("[heaps] %s FAILED (size %#llx) from lr %llx\n", what, (unsigned long long)size, (unsigned long long)lr);
}

#define CALL(sym)                                    \
    do {                                             \
        static uint64_t a_;                          \
        uint64_t lr_ = c->x[30];                     \
        guest_call(c, orig(sym, &a_));               \
        c->x[30] = lr_;                              \
    } while (0)

// Unit heaps are fixed pools (e.g. 256 x 0x50-byte objects per loaded area: bushsanity's extra actors ran out at
// ~450 actors around Link). Every unit heap gets UNIT_SCALE times the units in a buffer of ours (the game's own
// buffer stays unused); freed again in DestroyUnitHeap. LARECOMP_UNIT_HEAP_SCALE overrides (1 = the game's sizes).
typedef struct { uint64_t head; void *mem; } UnitBuf;
static UnitBuf g_unit[256];
static pthread_mutex_t g_unit_lock = PTHREAD_MUTEX_INITIALIZER;
#ifdef LARECOMP_COMPACT_GUEST
static void *g_resource_cache;
static uint64_t g_resource_allocator;
static size_t g_resource_size;
static _Atomic uint64_t g_resource_flush_time;

static void resource_cache_flush(uint64_t allocator) {
    if (allocator != g_resource_allocator || !g_resource_cache) return;
    uint64_t now = plat_now_ns(), last = atomic_load(&g_resource_flush_time);
    if (now - last >= 2000000000ull && atomic_compare_exchange_strong(&g_resource_flush_time, &last, now))
        msync(g_resource_cache, g_resource_size, MS_ASYNC);
}
#endif

// Reboot: our unit-heap buffers (host memory, not in the guest window) go with the game that made them.
RT_ON_RESET(heaps_reset) {
#ifdef LARECOMP_COMPACT_GUEST
    if (g_resource_cache) munmap(g_resource_cache, g_resource_size);
    g_resource_cache = NULL;
    g_resource_allocator = 0;
    g_resource_size = 0;
    atomic_store(&g_resource_flush_time, 0);
#endif
    for (int i = 0; i < 256; i++)
        if (g_unit[i].mem) {
#ifdef _WIN32
            _aligned_free(g_unit[i].mem);
#else
            free(g_unit[i].mem);
#endif
            g_unit[i].mem = NULL;
        }
}

static int unit_scale(void) {
    static int s = -1;
    if (s < 0) {
        const char *e = ENV("LARECOMP_UNIT_HEAP_SCALE");
        // Keep the game's original pools on mobile. The desktop's fourfold expansion eagerly zeros
        // extra backing memory even in vanilla play; large randomizer layouts can opt in explicitly.
#if defined(__APPLE__) && TARGET_OS_IPHONE
        s = e ? atoi(e) : 1;
#else
        s = e ? atoi(e) : 4;
#endif
        if (s < 1) s = 1;
    }
    return s;
}

// nn::lmem::CreateUnitHeap(void *addr, size_t size, size_t unit, int align, int opt, HeapHead *head)
HLE_FUNC("_ZN2nn4lmem14CreateUnitHeapEPvmmiiPNS0_6detail8HeapHeadE", heap_create_unit) {
    uint64_t lr = c->x[30], size = c->x[1], unit = c->x[2];
    int k = unit_scale();
    void *mem = NULL;
    if (k > 1 && unit) {
        uint64_t big = size * (uint64_t)k;
#ifdef _WIN32
        mem = _aligned_malloc(big, 0x1000);
        if (mem) {
#else
        if (posix_memalign(&mem, 0x1000, big) == 0) {
#endif
            memset(mem, 0, big);
            c->x[0] = (uint64_t)(uintptr_t)mem;
            c->x[1] = big;
        } else {
            mem = NULL;
        }
    }
    uint64_t head = c->x[5];
    CALL("_ZN2nn4lmem14CreateUnitHeapEPvmmiiPNS0_6detail8HeapHeadE");
    if (mem) {
        pthread_mutex_lock(&g_unit_lock);
        for (int i = 0; i < 256; i++)
            if (!g_unit[i].mem) { g_unit[i].head = c->x[0] ? c->x[0] : head; g_unit[i].mem = mem; break; }
        pthread_mutex_unlock(&g_unit_lock);
    }
    if (ENV("LARECOMP_LOG_HEAPS"))
        LOG("[heaps] unit heap %#llx bytes, unit %#llx (%llu units, x%d) from lr %llx -> %llx\n", (unsigned long long)size,
            (unsigned long long)unit, unit ? (unsigned long long)(size / unit) : 0ull, mem ? k : 1, (unsigned long long)lr,
            (unsigned long long)c->x[0]);
}

HLE_FUNC("_ZN2nn4lmem15DestroyUnitHeapEPNS0_6detail8HeapHeadE", heap_destroy_unit) {
    uint64_t head = c->x[0];
    CALL("_ZN2nn4lmem15DestroyUnitHeapEPNS0_6detail8HeapHeadE");
    pthread_mutex_lock(&g_unit_lock);
    for (int i = 0; i < 256; i++)
        if (g_unit[i].mem && g_unit[i].head == head) {
#ifdef _WIN32
            _aligned_free(g_unit[i].mem);
#else
            free(g_unit[i].mem);
#endif
            g_unit[i].mem = NULL;
            break;
        }
    pthread_mutex_unlock(&g_unit_lock);
}

HLE_FUNC("_ZN2nn4lmem20AllocateFromUnitHeapEPNS0_6detail8HeapHeadE", heap_alloc_unit) {
    uint64_t lr = c->x[30], head = c->x[0];
    CALL("_ZN2nn4lmem20AllocateFromUnitHeapEPNS0_6detail8HeapHeadE");
    if (!c->x[0]) fail_once("unit heap allocation", lr, head);
}

// nn::lmem::CreateFrameHeap(void *addr, size_t size, int opt)
HLE_FUNC("_ZN2nn4lmem15CreateFrameHeapEPvmi", heap_create_frame) {
    uint64_t lr = c->x[30], size = c->x[1];
    CALL("_ZN2nn4lmem15CreateFrameHeapEPvmi");
    if (ENV("LARECOMP_LOG_HEAPS"))
        LOG("[heaps] frame heap %#llx bytes from lr %llx\n", (unsigned long long)size, (unsigned long long)lr);
}

HLE_FUNC("_ZN2nn4lmem21AllocateFromFrameHeapEPNS0_6detail8HeapHeadEmi", heap_alloc_frame) {
    uint64_t lr = c->x[30], size = c->x[1];
    CALL("_ZN2nn4lmem21AllocateFromFrameHeapEPNS0_6detail8HeapHeadEmi");
    if (!c->x[0]) fail_once("frame heap allocation", lr, size);
}

// nn::mem::StandardAllocator::Initialize(void *addr, size_t size[, bool])
HLE_FUNC("_ZN2nn3mem17StandardAllocator10InitializeEPvm", heap_std_init) {
#ifdef LARECOMP_COMPACT_GUEST
    // v1.0.0 allocates its resource-cache buffer from the remaining application heap (+0x38 in the
    // cache owner), but initializes the allocator with a fixed 1334 MB even when that buffer is smaller.
    // The compact heap leaves only ~230 MB for that buffer, which cannot hold the title's resources.
    // The overworld loads a ~258 MB archive plus many other resources alongside the common cache.
    // Both 512 and 768 MB exhaust this allocator during that transition. Preserve the game's capacity
    // with temporary shared file backing: inactive archive pages can be reclaimed under memory
    // pressure instead of counting the entire working cache against the iPad's anonymous-memory limit.
    // GPU buffer mappings may also reference this cache. The caller restores x19 before this tail
    // call. Release the mapping only after guest threads and GPU commands stop on reboot.
    if (c->x[30] == MAIN_BASE + 0xa548 && c->x[2] == 0x53600000 &&
        c->x[19] >= HEAP_BASE && c->x[19] < HEAP_BASE + HEAP_MAX - 0x40) {
        uint64_t actual = LD64(c->x[19] + 0x38);
        if (!actual || actual > c->x[2]) FATAL("invalid compact resource-cache buffer size %llx\n", actual);
        const uint64_t size = c->x[2];
        if (!g_resource_cache) {
            g_resource_cache = resource_cache_map(size);
            if (!g_resource_cache) FATAL("cannot map compact resource cache: %s\n", strerror(errno));
            g_resource_size = size;
        }
        g_resource_allocator = c->x[0];
        LOG("[heaps] resource cache: %llu MB temporary file backing (guest remainder %llu MB)\n",
            (unsigned long long)(size >> 20), (unsigned long long)(actual >> 20));
        c->x[1] = (uint64_t)(uintptr_t)g_resource_cache;
        c->x[2] = size;
    }
#endif
    uint64_t lr = c->x[30], size = c->x[2];
    CALL("_ZN2nn3mem17StandardAllocator10InitializeEPvm");
    if (ENV("LARECOMP_LOG_HEAPS"))
        LOG("[heaps] standard allocator %#llx bytes from lr %llx\n", (unsigned long long)size, (unsigned long long)lr);
}

HLE_FUNC("_ZN2nn3mem17StandardAllocator10InitializeEPvmb", heap_std_init_b) {
    uint64_t lr = c->x[30], size = c->x[2];
    CALL("_ZN2nn3mem17StandardAllocator10InitializeEPvmb");
    if (ENV("LARECOMP_LOG_HEAPS"))
        LOG("[heaps] standard allocator %#llx bytes (b) from lr %llx\n", (unsigned long long)size, (unsigned long long)lr);
}

HLE_FUNC("_ZN2nn3mem17StandardAllocator8AllocateEm", heap_std_alloc) {
    uint64_t lr = c->x[30], size = c->x[1];
#ifdef LARECOMP_COMPACT_GUEST
    uint64_t allocator = c->x[0];
#endif
    CALL("_ZN2nn3mem17StandardAllocator8AllocateEm");
#ifdef LARECOMP_COMPACT_GUEST
    resource_cache_flush(allocator);
#endif
    if (!c->x[0] && size) fail_once("standard allocation", lr, size);
}

HLE_FUNC("_ZN2nn3mem17StandardAllocator8AllocateEmm", heap_std_alloc_a) {
    uint64_t lr = c->x[30], size = c->x[1];
#ifdef LARECOMP_COMPACT_GUEST
    uint64_t allocator = c->x[0];
#endif
    CALL("_ZN2nn3mem17StandardAllocator8AllocateEmm");
#ifdef LARECOMP_COMPACT_GUEST
    resource_cache_flush(allocator);
    if ((size >= (64ull << 20) || (!c->x[0] && size)) && ENV("LARECOMP_LOG_HEAPS"))
        LOG("[heaps] large allocation %llu MB, allocator %llx (%s) -> %llx, lr %llx\n",
            (unsigned long long)(size >> 20), (unsigned long long)allocator,
            allocator == g_resource_allocator ? "resource cache" : "other",
            (unsigned long long)c->x[0], (unsigned long long)lr);
#endif
    if (!c->x[0] && size) fail_once("standard allocation (aligned)", lr, size);
}
