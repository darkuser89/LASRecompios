// Bounded background compiler work. Snapshots and Metal compiler memory stay bounded together;
// graphics retries admission on a later draw, compute can require admission to preserve GPU work.
#pragma once
#import <Foundation/Foundation.h>
#include <stdatomic.h>

static inline unsigned metal_shader_workers(uint64_t memory, unsigned cpus) {
    return memory <= (4ull << 30) || cpus < 4 ? 1 : 2;
}

@interface MetalShaderJobs : NSObject {
@public
    _Atomic unsigned pending, running, peak, deferred, generation;
    unsigned workers, capacity;
@private
    NSOperationQueue *queue;
    NSCondition *admission;
}
- (instancetype)initWithWorkers:(unsigned)count;
- (NSOperation *)enqueue:(void (^)(void))work required:(BOOL)required;
- (void)reset;
@end

@implementation MetalShaderJobs
- (instancetype)initWithWorkers:(unsigned)count {
    if ((self = [super init])) {
        workers = MAX(1u, MIN(2u, count)); capacity = workers * 8;
        queue = [NSOperationQueue new];
        queue.name = @"larecomp.shaders";
        queue.qualityOfService = NSQualityOfServiceUserInitiated;
        queue.maxConcurrentOperationCount = workers;
        admission = [NSCondition new];
    }
    return self;
}
- (NSOperation *)enqueue:(void (^)(void))work required:(BOOL)required {
    [admission lock];
    while (atomic_load(&pending) >= capacity) {
        if (!required) { atomic_fetch_add(&deferred, 1); [admission unlock]; return nil; }
        [admission wait];
    }
    unsigned epoch = atomic_load(&generation);
    atomic_fetch_add(&pending, 1);
    NSBlockOperation *op = [NSBlockOperation blockOperationWithBlock:^{ @autoreleasepool {
        BOOL current = epoch == atomic_load(&self->generation);
        if (current) {
            unsigned n = atomic_fetch_add(&self->running, 1) + 1;
            unsigned high = atomic_load(&self->peak);
            while (n > high && !atomic_compare_exchange_weak(&self->peak, &high, n)) {}
            work();
            atomic_fetch_sub(&self->running, 1);
        }
        [self->admission lock];
        atomic_fetch_sub(&self->pending, 1);
        [self->admission broadcast];
        [self->admission unlock];
    } }];
    op.queuePriority = required ? NSOperationQueuePriorityVeryHigh : NSOperationQueuePriorityNormal;
    [queue addOperation:op];
    [admission unlock];
    return op;
}
- (void)reset {
    // Call on the submitting render thread after guest threads stop. Cancel queued generations,
    // drain at most `workers` active compiles, then release caches / unmap guest memory safely.
    atomic_fetch_add(&generation, 1);
    [queue waitUntilAllOperationsAreFinished];
}
@end
