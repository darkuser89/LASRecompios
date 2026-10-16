// Exercise the production texture lifecycle with real Metal allocations, uploads and readback.
// Unused UI/game entry points are removed by -dead_strip; patching and renderer view teardown are stubbed.
#include <assert.h>
#include "../runtime/nvn/metal.m"

int texpatch_identify(const NvnTexture *t) { (void)t; return -1; }
uint32_t texpatch_generation(void) { return 1; }
void texpatch_apply(int id, uint32_t level, uint8_t *data, uint32_t w, uint32_t h, uint32_t row) {
    (void)id; (void)level; (void)data; (void)w; (void)h; (void)row;
}
void renderer_texture_gone(void *nvn, void *mtl) { (void)nvn; (void)mtl; }
void rt_fatal_trace(void) { abort(); }

static void check_pixels(NvnTexture *t, uint8_t value) {
    id<MTLTexture> tex=renderer_texture(t,nil,false);
    id<MTLBuffer> readback=[dev newBufferWithLength:256*256*4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLBlitCommandEncoder> bl=[cb blitCommandEncoder];
    [bl copyFromTexture:tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
            sourceSize:MTLSizeMake(256,256,1) toBuffer:readback destinationOffset:0
            destinationBytesPerRow:1024 destinationBytesPerImage:256*256*4];
    [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
    assert(cb.status==MTLCommandBufferStatusCompleted);
    for(unsigned i=0;i<256*256*4;i++)assert(((uint8_t *)readback.contents)[i]==value);
}

int main(void) { @autoreleasepool {
    setenv("LARECOMP_TEXTURE_CACHE_MB","1",1);
    dev=MTLCreateSystemDefaultDevice(); assert(dev);
    queue=[dev newCommandQueue];
    NvnTexture textures[10]={0};
    for(unsigned i=0;i<10;i++) {
        NvnTexture *t=&textures[i];
        t->target=1; t->format=0x25; t->width=t->height=256; t->depth=t->levels=1;
        t->layout=(TexLayout){.width=256,.height=256,.depth=1,.layers=1,.levels=1,.fmt=nvn_format_info(0x25)};
        tegra_layout_init(&t->layout);
        uint64_t size=tegra_total_size(&t->layout);
        uint8_t *data=malloc(size); assert(data); memset(data,11+i,size); t->addr=(uintptr_t)data;
        check_pixels(t,11+i);
    }
    // Four kinds of irreplaceable/current state must remain allocated even above the soft budget.
    ((TexBackend *)textures[0].backend)->gpu_owned=true;
    ((TexBackend *)textures[1].backend)->dynamic=true;
    ((TexBackend *)textures[2].backend)->uploaded=false;
    ((TexBackend *)textures[3].backend)->used_frame=590;
    total_frames=1200;
    ((TexBackend *)textures[3].backend)->used_frame=1190;
    trim_texture_cache();
    for(unsigned i=0;i<4;i++)assert(((TexBackend *)textures[i].backend)->tex);
    unsigned evicted=0;
    for(unsigned i=4;i<10;i++)evicted+=((TexBackend *)textures[i].backend)->tex==nil;
    assert(evicted>0 && g_texture_evictions==evicted);
    uint64_t bytes=g_tex_live_bytes;
    for(unsigned i=0;i<10;i++)check_pixels(&textures[i],11+i);
    assert(g_texture_reloads==evicted && g_tex_live_bytes>bytes);
    // Repeated eviction/restoration and destroying an already evicted object cannot double release.
    total_frames=2400;
    ((TexBackend *)textures[3].backend)->used_frame=2390;
    trim_texture_cache();
    for(unsigned i=0;i<10;i++){backend_texture_destroyed(&textures[i]);free((void *)(uintptr_t)textures[i].addr);}
    assert(g_tex_live==0 && g_tex_live_bytes==0 && g_reg_n==0);
    printf("Texture cache: %u cold copies dropped, byte-exact GPU restore, protected resources, repeat eviction and teardown passed\n",evicted);
} }
