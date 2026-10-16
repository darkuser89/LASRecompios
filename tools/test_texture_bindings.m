// Check actual GPU pixels while texture/sampler bindings change, repeat and cross encoder boundaries.
#import <Foundation/Foundation.h>
#include <assert.h>
#include <stdio.h>
#include "../runtime/nvn/texture_bindings.h"

static const char *source =
"#include <metal_stdlib>\nusing namespace metal;\n"
"struct V { float4 p [[position]]; uint x [[flat]]; };\n"
"vertex V vs(uint i [[vertex_id]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
"const float2 p[]={float2(-1,-1),float2(3,-1),float2(-1,3)};\n"
"return {float4(p[i],0,1),uint(round(t.sample(s,float2(0.5),level(0)).r*255))}; }\n"
"fragment uint fs(V v [[stage_in]],texture2d<float> t [[texture(0)]],sampler s [[sampler(0)]]) {\n"
"return v.x+uint(round(t.sample(s,float2(0.5)).r*255)); }\n";

int main(void) { @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice(); assert(d);
    NSError *error = nil;
    id<MTLLibrary> lib = [d newLibraryWithSource:@(source) options:nil error:&error];
    if (!lib) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
    assert(lib);
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
    id<MTLRenderPipelineState> pipe = [d newRenderPipelineStateWithDescriptor:pd error:&error]; assert(pipe);
    id<MTLCommandQueue> q = [d newCommandQueue];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:2 height:1 mipmapped:NO];
    td.storageMode = MTLStorageModeShared; td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> a = [d newTextureWithDescriptor:td], b = [d newTextureWithDescriptor:td];
    uint8_t av[] = {10,0,0,255,210,0,0,255}, bv[] = {20,0,0,255,60,0,0,255};
    [a replaceRegion:MTLRegionMake2D(0,0,2,1) mipmapLevel:0 withBytes:av bytesPerRow:8];
    [b replaceRegion:MTLRegionMake2D(0,0,2,1) mipmapLevel:0 withBytes:bv bytesPerRow:8];
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    id<MTLSamplerState> nearest = [d newSamplerStateWithDescriptor:sd];
    sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
    id<MTLSamplerState> linear = [d newSamplerStateWithDescriptor:sd];
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:6 height:1 mipmapped:NO];
    td.storageMode = MTLStorageModeShared; td.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> out = [d newTextureWithDescriptor:td];
    TextureBindings bindings = {0};
    uint64_t calls[2];
    for (unsigned enabled=0; enabled<2; enabled++) {
        texture_bindings_reset(&bindings); bindings.calls=bindings.requests=0;
        for (unsigned round=0; round<2; round++) { @autoreleasepool {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture=out;
            rp.colorAttachments[0].loadAction=MTLLoadActionClear;
            rp.colorAttachments[0].storeAction=MTLStoreActionStore;
            id<MTLRenderCommandEncoder> enc=[cb renderCommandEncoderWithDescriptor:rp];
            texture_bindings_begin(&bindings,enc);
            [enc setRenderPipelineState:pipe];
            for (unsigned i=0;i<6;i++) {
                texture_bindings_begin(&bindings,enc);
                id<MTLTexture> vt=i<3?a:b, ft=i<4?b:a;
                id<MTLSamplerState> vs=i<2?nearest:linear, fs=i<5?linear:nearest;
                texture_bindings_set(&bindings,enc,0,0,vt,vs,enabled);
                texture_bindings_set(&bindings,enc,1,0,ft,fs,enabled);
                // Duplicate state, high slot, and explicit nil must all be tracked independently.
                texture_bindings_set(&bindings,enc,0,0,vt,vs,enabled);
                texture_bindings_set(&bindings,enc,1,0,ft,fs,enabled);
                texture_bindings_set(&bindings,enc,1,15,nil,nil,enabled);
                [enc setViewport:(MTLViewport){i,0,1,1,0,1}];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            }
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if(cb.error)fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
            assert(cb.status==MTLCommandBufferStatusCompleted);
            uint32_t pixels[6], expected[]={250,250,150,80,150,250};
            [out getBytes:pixels bytesPerRow:sizeof pixels fromRegion:MTLRegionMake2D(0,0,6,1) mipmapLevel:0];
            for(unsigned i=0;i<6;i++) { if(pixels[i]!=expected[i])fprintf(stderr,"pixel %u: %u != %u\n",i,pixels[i],expected[i]); assert(pixels[i]==expected[i]); }
        } }
        calls[enabled]=bindings.calls;
        printf("%s: %llu requests, %llu Metal calls, pixels correct across both encoders\n",enabled?"cached":"legacy",(unsigned long long)bindings.requests,(unsigned long long)bindings.calls);
    }
    assert(calls[1]<calls[0]);
    puts("Texture/sampler state changes, duplicates, nils, stages and encoder lifetime passed");
} }
