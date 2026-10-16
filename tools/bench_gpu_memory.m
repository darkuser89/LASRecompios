// CPU cost of a scene's buffer lifetime: unique ranges, shared ranges and shuffled destruction.
// No GPU mapping is requested, so the synthetic addresses are never dereferenced.
#import <Foundation/Foundation.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../runtime/nvn/gpu_memory.h"

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
int main(void) { @autoreleasepool {
    enum { N = 32768 };
    unsigned *order = malloc(N * sizeof *order);
    assert(order);
    for (unsigned i = 0; i < N; i++) order[i] = i;
    uint32_t seed = 0x12345678;
    for (unsigned i = N - 1; i; i--) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        unsigned j = seed % (i + 1), swap = order[i]; order[i] = order[j]; order[j] = swap;
    }
    for (int round = 0; round < 3; round++) {
        double start = now_ms();
        for (unsigned i = 0; i < N; i++) {
            uint64_t address = 0x100000000ull + i * 0x4000ull;
            gpu_pool_add(address, 16);
            if (i % 7 == 0) gpu_pool_add(address, 16);
        }
        double added = now_ms();
        for (unsigned i = 0; i < N; i++) {
            unsigned k = order[i];
            uint64_t address = 0x100000000ull + k * 0x4000ull;
            gpu_pool_remove(address, 16);
            if (k % 7 == 0) gpu_pool_remove(address, 16);
        }
        double removed = now_ms();
        printf("round %d: %d unique buffers, add %.3f ms, shuffled removal %.3f ms\n",
               round + 1, N, added - start, removed - added);
        gpu_memory_reset();
    }
    free(order);
    return 0;
} }
