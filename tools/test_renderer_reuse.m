// Exercise production clear/preparation/view/index paths. Only the game's texture storage and HLE are stubbed;
// all state objects, copies, command encoders and readback use the real Metal device.
#include <assert.h>
#include "../runtime/nvn/renderer.m"

static id<MTLCommandQueue> test_queue;
static uint64_t test_frame;
static float test_scale=1;
id<MTLBuffer> g_guest_buffer;
id<MTLCommandQueue> renderer_queue(void) { return test_queue; }
uint64_t renderer_present_count(void) { return test_frame; }
id<MTLTexture> renderer_texture(NvnTexture *t, id<MTLBlitCommandEncoder> b, bool render) {
    (void)b; (void)render; return (__bridge id<MTLTexture>)t->backend;
}
void renderer_mark_gpu_owned(NvnTexture *t) { (void)t; }
bool renderer_texture_first_write(NvnTexture *t) { (void)t; return false; }
bool renderer_texture_frame_fresh(NvnTexture *t) { (void)t; return false; }
bool renderer_texture_level_fresh(NvnTexture *t) { (void)t; return false; }
void renderer_texture_scale(NvnTexture *t, float *x, float *y) { (void)t; *x=*y=test_scale; }
MTLPixelFormat renderer_pixel_format(uint32_t f) { return f==0x35?MTLPixelFormatDepth32Float_Stencil8:MTLPixelFormatRGBA8Unorm; }
bool renderer_format_is_depth(uint32_t f) { return f==0x35; }
bool renderer_format_has_stencil(uint32_t f) { return f==0x35; }
bool renderer_format_alpha_one(uint32_t f) { return f==0x19; }
int renderer_texpatch_of(NvnTexture *t) { (void)t; return -1; }
int texpatch_bottle_kind(int id) { (void)id; return 0; }
int texpatch_heart_kind(int id) { (void)id; return 0; }
bool texpatch_is_item_cursor(int id) { (void)id; return false; }
void renderer_save_texture(id<MTLTexture> t, const char *p) { (void)t; (void)p; }
void renderer_save_texture_alpha(id<MTLTexture> t, const char *p) { (void)t; (void)p; }
void rt_fatal_trace(void) { abort(); }

int main(void) { @autoreleasepool {
    dev=MTLCreateSystemDefaultDevice(); assert(dev); test_queue=[dev newCommandQueue]; renderer_init(dev);
    MTLTextureDescriptor *td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:1 mipmapped:NO];
    td.storageMode=MTLStorageModeShared; td.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
    id<MTLTexture> color=[dev newTextureWithDescriptor:td], other=[dev newTextureWithDescriptor:td];
    NvnTexture target={.width=4,.height=1,.format=0x25,.backend=(__bridge void *)color};
    st.rt.ncolor=1; st.rt.color[0]=&target; st.rt_dirty=true; st.scissor[0][2]=4; st.scissor[0][3]=1;
    const float red[4]={1,0,0,1}, green[4]={0,0.5,0,1};
    clear(0,0,red,15,false,0,false,0,0);
    clear(0,0,red,15,false,0,false,0,0);
    assert(next_clear_pipeline==1 && next_clear_depth==1);
    clear(0,0,green,2,false,0,false,0,0);
    assert(next_clear_pipeline==2 && next_clear_depth==1);
    renderer_flush();
    uint8_t pixels[16]; [color getBytes:pixels bytesPerRow:16 fromRegion:MTLRegionMake2D(0,0,4,1) mipmapLevel:0];
    for (unsigned i=0;i<4;i++) assert(pixels[4*i]==255 && abs((int)pixels[4*i+1]-128)<=1 && pixels[4*i+2]==0 && pixels[4*i+3]==255);
    assert(render_state.encoder==nil); // clear invalidated the draw cache; the next draw must rebind

    // MRT format belongs in the key even when clearing the same first attachment.
    NvnTexture second={.width=4,.height=1,.format=0x25,.backend=(__bridge void *)other};
    st.rt.ncolor=2; st.rt.color[1]=&second; st.rt_dirty=true;
    clear(0,0,red,15,false,0,false,0,0); renderer_flush(); assert(next_clear_pipeline==3);
    td.pixelFormat=MTLPixelFormatDepth32Float_Stencil8; td.storageMode=MTLStorageModePrivate;
    id<MTLTexture> depth=[dev newTextureWithDescriptor:td]; assert(depth);
    NvnTexture depth_target={.width=4,.height=1,.format=0x35,.backend=(__bridge void *)depth};
    st.rt.depth=&depth_target; st.rt_dirty=true;
    clear(-1,0,NULL,0,true,0.25,true,0x55,0xff);
    clear(-1,0,NULL,0,true,0.25,true,0x55,0xff);
    renderer_flush(); assert(next_clear_depth==2);
    clear(-1,0,NULL,0,true,0.5,true,0xaa,0x0f); renderer_flush(); assert(next_clear_depth==3);

    // View reuse and lifetime invalidation: the cache is not allowed to hold freed bases indefinitely.
    id<MTLTexture> v=renderer_cached_view(color,color.pixelFormat,MTLTextureType2DArray,NSMakeRange(0,1),NSMakeRange(0,1));
    assert(v!=color && v==renderer_cached_view(color,color.pixelFormat,MTLTextureType2DArray,NSMakeRange(0,1),NSMakeRange(0,1)));
    id<MTLTexture> nested=renderer_cached_view(v,v.pixelFormat,MTLTextureType2D,NSMakeRange(0,1),NSMakeRange(0,1));
    assert(nested!=v);
    int live=g_views_live; views_forget((__bridge void *)color); assert(g_views_live==live-2);
    assert(v!=renderer_cached_view(color,color.pixelFormat,MTLTextureType2DArray,NSMakeRange(0,1),NSMakeRange(0,1)));
    td.pixelFormat=MTLPixelFormatRGBA8Unorm; td.storageMode=MTLStorageModePrivate;
    td.textureType=MTLTextureType2DArray; td.arrayLength=512;
    id<MTLTexture> array=[dev newTextureWithDescriptor:td]; assert(array);
    id<MTLTexture> small=renderer_cached_view(array,array.pixelFormat,MTLTextureType2DArray,NSMakeRange(0,1),NSMakeRange(0,1));
    id<MTLTexture> large=renderer_cached_view(array,array.pixelFormat,MTLTextureType2DArray,NSMakeRange(0,1),NSMakeRange(0,257));
    assert(small!=large && small.arrayLength==1 && large.arrayLength==257);
    NvnTexture rgbx=target; rgbx.format=0x19;
    id<MTLTexture> rgbx_view=sampled_texture(&rgbx,NULL,false,TEXTYPE_2D);
    assert(rgbx_view!=color && rgbx_view.swizzle.alpha==MTLTextureSwizzleOne && color.swizzle.alpha==MTLTextureSwizzleAlpha);

    NvnTexEntry entries[2]={0}; entries[1].tex=&target;
    NvnDescPool pool={.count=2,.entries=entries}; st.tpool=&pool; st.spool=NULL; st.tex[1][0]=1;
    ShaderTexture shader={.handle_offset=0x20,.type=TEXTYPE_2D};
    PreparedBinding *prepared=prepare_binding(1,0,&shader); assert(prepared->texture==color);
    uint64_t hits=prepare_hits, misses=prepare_misses;
    assert(prepare_binding(1,0,&shader)->texture==color && prepare_hits==hits+1);
    entries[1].tex=&second; __atomic_add_fetch(&pool.generation,1,__ATOMIC_RELEASE);
    assert(prepare_binding(1,0,&shader)->texture==other && prepare_misses==misses+1);
    test_frame++; misses=prepare_misses;
    prepare_binding(1,0,&shader); assert(prepare_misses==misses+1);
    second.backend=(__bridge void *)color; renderer_texture_cache_changed();
    assert(prepare_binding(1,0,&shader)->texture==color);
    test_scale=0.5; renderer_texture_cache_changed();
    assert(prepare_binding(1,0,&shader)->scale[0]==0.5);
    // Re-registering a view must also invalidate preparation, not merely the Metal binding cache.
    entries[1].has_view=true; entries[1].view.setmask=8;
    entries[1].view.swizzle[0]=4; entries[1].view.swizzle[1]=3; entries[1].view.swizzle[2]=2; entries[1].view.swizzle[3]=5;
    __atomic_add_fetch(&pool.generation,1,__ATOMIC_RELEASE);
    assert(prepare_binding(1,0,&shader)->texture!=color);

    uint32_t n; const uint8_t quads[4]={3,2,1,0};
    id<MTLBuffer> indices=expand_indices(7,4,quads,0,0,&n); const uint32_t expected[]={3,2,1,3,1,0};
    assert(n==6 && !memcmp(indices.contents,expected,sizeof expected));
    indices=expand_indices(6,4,NULL,0,4,&n); const uint32_t fan[]={4,5,6,4,6,7};
    assert(n==6 && !memcmp(indices.contents,fan,sizeof fan));
    puts("Production clear: cache reuse/MRT/depth/stencil masks and pixels passed; prepared textures invalidate on frame, descriptor, storage and scale; cached views and quad/fan indices passed");
} return 0; }
