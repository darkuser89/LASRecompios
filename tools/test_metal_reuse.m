// Real GPU checks for encoder-local state, immutable snapshots, shared shader stages and concurrent compilation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <assert.h>
#include <stdio.h>
#include "../runtime/nvn/render_state.h"
#include "../runtime/nvn/shader_buffers.h"
#include "../runtime/nvn/shader_cache.h"

static void check_state(id<MTLDevice> dev, id<MTLCommandQueue> queue, bool legacy) {
    const char *source = "#include <metal_stdlib>\nusing namespace metal;\n"
        "vertex float4 vs(uint i [[vertex_id]]) { float2 p[]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[i],0,1);}\n"
        "fragment uint fs(constant uint &v [[buffer(0)]]) {return v;}\n";
    NSError *error = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(source) options:nil error:&error]; assert(lib);
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"vs"]; pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
    id<MTLRenderPipelineState> pipe = [dev newRenderPipelineStateWithDescriptor:pd error:&error]; assert(pipe);
    MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new]; dd.depthCompareFunction = MTLCompareFunctionAlways;
    id<MTLDepthStencilState> depth = [dev newDepthStencilStateWithDescriptor:dd];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:4 height:1 mipmapped:NO];
    td.storageMode = MTLStorageModeShared; td.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
    RenderState state = {.legacy=legacy};
    for (unsigned round = 0; round < 2; round++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = tex; rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
        render_state_begin(&state,e);
        for (unsigned i = 0; i < 4; i++) {
            if (i == 2) {
                // A clear/other direct writer can change the encoder behind the cache.
                [e setCullMode:MTLCullModeFront]; render_state_reset(&state); render_state_begin(&state,e);
            }
            for (unsigned repeat=0;repeat<3;repeat++) {
                render_state_pipeline(&state,e,pipe); render_state_depth(&state,e,depth);
                render_state_cull(&state,e,MTLCullModeNone); render_state_winding(&state,e,MTLWindingCounterClockwise);
                render_state_fill(&state,e,MTLTriangleFillModeFill); render_state_stencil(&state,e,0,0);
                const float blend[4]={0,0,0,0}, bias[3]={0,0,0};
                render_state_blend(&state,e,blend); render_state_bias(&state,e,bias);
                render_state_viewport(&state,e,(MTLViewport){i,0,1,1,0,1});
                render_state_scissor(&state,e,(MTLScissorRect){i,0,1,1});
            }
            uint32_t value=11+i+round*10; [e setFragmentBytes:&value length:4 atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; assert(!cb.error);
        uint32_t pixels[4]; [tex getBytes:pixels bytesPerRow:16 fromRegion:MTLRegionMake2D(0,0,4,1) mipmapLevel:0];
        for (unsigned i=0;i<4;i++) assert(pixels[i]==11+i+round*10);
    }
    assert(legacy ? state.calls==state.requests : state.calls<state.requests/2);
    printf("Render state %s: %llu requests -> %llu calls, pixels correct after changes, reset and new encoder\n",
        legacy?"legacy":"cached",(unsigned long long)state.requests,(unsigned long long)state.calls);
}

int main(void) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); assert(dev);
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    check_state(dev,queue,false); check_state(dev,queue,true);
    ShaderBuffers snapshots={0};
    uint32_t data[4]={11,22,33,44};
    id<MTLBuffer> a=shader_constant_buffer(&snapshots,dev,(uintptr_t)data,sizeof data);
    assert(a && a==shader_constant_buffer(&snapshots,dev,(uintptr_t)data,sizeof data));
    data[0]=99;
    id<MTLBuffer> b=shader_constant_buffer(&snapshots,dev,(uintptr_t)data,sizeof data);
    assert(b && a!=b && *(uint32_t *)a.contents==11 && *(uint32_t *)b.contents==99);
    assert(b==shader_constant_buffer(&snapshots,dev,(uintptr_t)data,sizeof data));
    assert(snapshots.copies==2 && snapshots.requests==4);

    MetalShaderCache shaders={0};
    MetalShaderCache *shared_shaders=&shaders;
    const char *source="#include <metal_stdlib>\nusing namespace metal; kernel void main0(constant uint *a [[buffer(0)]],constant uint *b [[buffer(1)]],device uint *out [[buffer(2)]]) {out[0]=a[0];out[1]=b[0];}\n";
    __block id<MTLFunction> function;
    dispatch_apply(24,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), ^(size_t i) { @autoreleasepool {
        NSString *error=nil; id<MTLFunction> f=metal_shader_function(shared_shaders,dev,source,false,&error); assert(f && !error);
        @synchronized(dev) { if (!function) function=f; else assert(function==f); }
    } });
    assert(shaders.compilations==1 && shaders.compilation_hits==23);
    NSError *error=nil; id<MTLComputePipelineState> pipe=[dev newComputePipelineStateWithFunction:function error:&error]; assert(pipe);
    id<MTLBuffer> output=[dev newBufferWithLength:8 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb=[queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:pipe]; [e setBuffer:a offset:0 atIndex:0]; [e setBuffer:b offset:0 atIndex:1]; [e setBuffer:output offset:0 atIndex:2];
    [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e endEncoding]; [cb commit];
    shader_buffers_reset(&snapshots); [cb waitUntilCompleted]; assert(!cb.error);
    assert(((uint32_t *)output.contents)[0]==11 && ((uint32_t *)output.contents)[1]==99);

    uint64_t vertex[19]={0}; vertex[17]=0x50b0000000070f00ull; vertex[18]=0xe30000000007000full;
    ShaderKey key={0};
    MetalShaderPart *part=metal_shader_part(&shaders,(uintptr_t)vertex,1,sizeof vertex,STAGE_VERTEX,&key); assert(part->info->msl);
    key.rt_kind[0]=2; key.vs_out_mask=0xffff; // irrelevant to vertex translation, e.g. a different fragment pairing
    assert(part==metal_shader_part(&shaders,(uintptr_t)vertex,1,sizeof vertex,STAGE_VERTEX,&key));
    key.flip_x=true;
    assert(part!=metal_shader_part(&shaders,(uintptr_t)vertex,1,sizeof vertex,STAGE_VERTEX,&key));
    assert(part!=metal_shader_part(&shaders,(uintptr_t)vertex,2,sizeof vertex,STAGE_VERTEX,&key)); // recycled code address
    // Cached canonical keys must produce exactly the same source as the direct translator.
    ((uint32_t *)vertex)[0x78/4]=15;
    vertex[17]=0xefd800000007ff00ull | (0x80ull<<20); // LD attribute 0: vertex kind / fragment varying mask matter
    vertex[18]=0xe30000000007000full;
    for (unsigned stage=0;stage<2;stage++) for (unsigned variant=0;variant<16;variant++) {
        ShaderKey k={0}; k.flip_x=variant&1; k.flip_y=variant&2; k.depth_minus_one=variant&4;
        k.attr_kind[0]=(variant>>2)%3; k.rt_kind[0]=(variant>>1)%3; k.vs_out_mask=variant&8?0xffff:0;
        ShaderInfo *direct=maxwell_translate((uint8_t *)vertex,sizeof vertex,stage,&k);
        MetalShaderPart *cached=metal_shader_part(&shaders,(uintptr_t)vertex,3,sizeof vertex,stage,&k);
        assert((direct->msl!=NULL)==(cached->info->msl!=NULL));
        if (direct->msl) assert(!strcmp(direct->msl,cached->info->msl));
        else assert(!strcmp(direct->error,cached->info->error));
        maxwell_free(direct);
    }
    metal_shader_cache_reset(&shaders);
    assert(metal_shader_function(&shaders,dev,source,false,NULL)==function);
    puts("Constants: immutable reuse, changed bytes/address, in-flight teardown passed; shader stages/24 concurrent compiles shared correctly");
} return 0; }
