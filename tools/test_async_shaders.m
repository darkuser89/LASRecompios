// Production scheduler, shader snapshot ownership and compute ordering on a real Metal device.
#define main renderer_reuse_checks
#include "test_renderer_reuse.m"
#undef main
#include <unistd.h>

static void wait_jobs(MetalShaderJobs *jobs) {
    double deadline = now_ms() + 30000;
    while (atomic_load(&jobs->pending)) { assert(now_ms() < deadline); usleep(1000); }
}

static void check_queue(unsigned workers) {
    MetalShaderJobs *jobs = [[MetalShaderJobs alloc] initWithWorkers:workers];
    dispatch_semaphore_t entered = dispatch_semaphore_create(0), release = dispatch_semaphore_create(0);
    __block _Atomic unsigned executed = 0;
    for (unsigned i = 0; i < workers; i++) assert([jobs enqueue:^{
        assert(!NSThread.isMainThread); atomic_fetch_add(&executed, 1);
        dispatch_semaphore_signal(entered); dispatch_semaphore_wait(release, DISPATCH_TIME_FOREVER);
    } required:NO]);
    for (unsigned i = 0; i < workers; i++) dispatch_semaphore_wait(entered, DISPATCH_TIME_FOREVER);
    for (unsigned i = workers; i < jobs->capacity; i++) assert([jobs enqueue:^{ atomic_fetch_add(&executed, 1); } required:NO]);
    assert(![jobs enqueue:^{ abort(); } required:NO]);
    assert(atomic_load(&jobs->pending) == jobs->capacity && atomic_load(&jobs->peak) == workers);
    // Reboot cancels queued work before it can touch old caches; active jobs drain.
    dispatch_group_t resetting = dispatch_group_create();
    dispatch_group_async(resetting, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{ [jobs reset]; });
    double deadline = now_ms() + 30000;
    while (!atomic_load(&jobs->generation)) { assert(now_ms() < deadline); usleep(1000); }
    for (unsigned i = 0; i < workers; i++) dispatch_semaphore_signal(release);
    dispatch_group_wait(resetting, DISPATCH_TIME_FOREVER);
    assert(!atomic_load(&jobs->pending) && !atomic_load(&jobs->running));
    assert(atomic_load(&executed) == workers);
    NSOperation *next = [jobs enqueue:^{ atomic_fetch_add(&executed, 1); } required:YES];
    [next waitUntilFinished]; assert(atomic_load(&executed) == workers + 1);
    printf("Scheduler %u worker(s): bounded admission, background execution, reset cancellation and reuse passed\n", workers);
}

static void check_snapshots(void) {
    // Hold every worker so input bytes can be destroyed before translation starts.
    dispatch_semaphore_t entered = dispatch_semaphore_create(0), release = dispatch_semaphore_create(0);
    for (unsigned i = 0; i < shader_jobs->workers; i++) [shader_jobs enqueue:^{
        dispatch_semaphore_signal(entered); dispatch_semaphore_wait(release, DISPATCH_TIME_FOREVER);
    } required:NO];
    for (unsigned i = 0; i < shader_jobs->workers; i++) dispatch_semaphore_wait(entered, DISPATCH_TIME_FOREVER);
    uint64_t vertex[0x230 / 8 + 1] = {0}, fragment[0x230 / 8 + 1] = {0};
    vertex[17] = fragment[17] = 0x50b0000000070f00ull;
    vertex[18] = fragment[18] = 0xe30000000007000full;
    NvnProgram p = {.stage_mask = 3};
    p.stages[0].code_addr = (uintptr_t)vertex; p.stages[1].code_addr = (uintptr_t)fragment;
    p.stages[0].const_offset = p.stages[1].const_offset = 19 * 8;
    p.stages[0].code_hash = 123; p.stages[1].code_hash = 456;
    ShaderKey vk = {0}, fk = {0};
    Prog *program = get_program(&p, &vk, &fk); assert(program && !program->ready && !program->vi);
    assert(program == get_program(&p, &vk, &fk));
    memset(vertex, 0xff, sizeof vertex); memset(fragment, 0xff, sizeof fragment); memset(&p, 0, sizeof p);
    for (unsigned i = 0; i < shader_jobs->workers; i++) dispatch_semaphore_signal(release);
    wait_jobs(shader_jobs); assert(program->ready && !program->failed && program->vs && program->fs);

    uint64_t compute[4] = {0, 0x50b0000000070f00ull, 0xe30000000007000full, 0x50b0000000070f00ull};
    NvnProgram cp = {.stage_mask = 1u << 5};
    cp.stages[5].code_addr = (uintptr_t)compute; cp.stages[5].const_offset = sizeof compute;
    cp.stages[5].code_hash = 789;
    (void)compute_program(&cp, false);
    memset(compute, 0xff, sizeof compute);
    CsProg *ready = compute_program(&cp, true); assert(ready && ready->pipe && ready->ready && !ready->failed);
    id<MTLCommandBuffer> cb = [test_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ready->pipe];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; assert(!cb.error);
    renderer_reset(); assert(!renderer_pending_shaders() && !g_cs_progs.count && !cache.count);
    puts("Production graphics/compute: immutable input, coalescing, async compile, required compute dispatch and reboot passed");
}

static void check_required_admission(void) {
    MetalShaderJobs *jobs = [[MetalShaderJobs alloc] initWithWorkers:1];
    dispatch_semaphore_t entered = dispatch_semaphore_create(0), release = dispatch_semaphore_create(0);
    [jobs enqueue:^{ dispatch_semaphore_signal(entered); dispatch_semaphore_wait(release, DISPATCH_TIME_FOREVER); } required:NO];
    dispatch_semaphore_wait(entered, DISPATCH_TIME_FOREVER);
    for (unsigned i = 1; i < jobs->capacity; i++) assert([jobs enqueue:^{} required:NO]);
    __block _Atomic bool submitting = false, executed = false;
    dispatch_group_t required = dispatch_group_create();
    dispatch_group_async(required, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        atomic_store(&submitting, true);
        NSOperation *op = [jobs enqueue:^{ assert(!NSThread.isMainThread); atomic_store(&executed, true); } required:YES];
        assert(op); [op waitUntilFinished];
    });
    double deadline = now_ms() + 30000;
    while (!atomic_load(&submitting)) { assert(now_ms() < deadline); usleep(1000); }
    assert(!atomic_load(&executed) && atomic_load(&jobs->pending) == jobs->capacity);
    dispatch_semaphore_signal(release);
    assert(!dispatch_group_wait(required, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC)));
    wait_jobs(jobs); assert(atomic_load(&executed));
    puts("Full queue: required work waits for bounded admission and executes exactly once");
}

static void check_reset_pending(void) {
    dispatch_semaphore_t entered = dispatch_semaphore_create(0), release = dispatch_semaphore_create(0);
    for (unsigned i = 0; i < shader_jobs->workers; i++) [shader_jobs enqueue:^{
        dispatch_semaphore_signal(entered); dispatch_semaphore_wait(release, DISPATCH_TIME_FOREVER);
    } required:NO];
    for (unsigned i = 0; i < shader_jobs->workers; i++) dispatch_semaphore_wait(entered, DISPATCH_TIME_FOREVER);
    uint64_t code[0x230 / 8 + 1] = {0}; code[17] = 0x50b0000000070f00ull; code[18] = 0xe30000000007000full;
    NvnProgram p = {.stage_mask = 3};
    for (int i = 0; i < 2; i++) { p.stages[i].code_addr = (uintptr_t)code; p.stages[i].const_offset = 19 * 8; }
    ShaderKey key = {0}; Prog *old = get_program(&p, &key, &key); assert(old && !old->ready);
    unsigned epoch = atomic_load(&shader_jobs->generation);
    dispatch_group_t gate = dispatch_group_create();
    dispatch_group_async(gate, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        double deadline = now_ms() + 30000;
        while (atomic_load(&shader_jobs->generation) == epoch) { assert(now_ms() < deadline); usleep(1000); }
        for (unsigned i = 0; i < shader_jobs->workers; i++) dispatch_semaphore_signal(release);
    });
    renderer_reset(); dispatch_group_wait(gate, DISPATCH_TIME_FOREVER);
    assert(!old->ready && !renderer_pending_shaders() && !cache.count);
    Prog *fresh = get_program(&p, &key, &key); assert(fresh && fresh != old);
    wait_jobs(shader_jobs); assert(fresh->ready && !fresh->failed);
    renderer_reset();
    puts("Production reboot with pending preparation: queued shader canceled, same-address program rebuilt safely");
}

static void check_debug_identity(void) {
    uint8_t code[0x230] = {0}, snapshot[sizeof code];
    ((uint64_t *)code)[17] = 0x50b0000000070f00ull;
    ((uint64_t *)code)[18] = 0xe30000000007000full;
    memcpy(snapshot, code, sizeof code);
    ShaderKey key = {0};
    for (int stage = 0; stage < 2; stage++) {
        maxwell_dbg_vs = maxwell_dbg_code = code; maxwell_dbg_pc = 0xffffffffu;
        ShaderInfo *direct = maxwell_translate(code, 19 * 8, stage, &key);
        ShaderInfo *copy = maxwell_translate_snapshot(snapshot, 19 * 8, stage, &key, (uintptr_t)code);
        assert(direct->msl && copy->msl && !strcmp(direct->msl, copy->msl));
        maxwell_free(direct); maxwell_free(copy);
    }
    maxwell_dbg_vs = maxwell_dbg_code = NULL;
    puts("Vertex/fragment debug probes preserve guest identity with owned snapshots");
}

int main(void) { @autoreleasepool {
    assert(metal_shader_workers(4ull << 30, 6) == 1);
    assert(metal_shader_workers(8ull << 30, 6) == 2);
    assert(metal_shader_workers(8ull << 30, 2) == 1);
    check_queue(1); check_queue(2); check_required_admission();
    renderer_reuse_checks(); check_snapshots(); check_reset_pending(); check_debug_identity();
} return 0; }
