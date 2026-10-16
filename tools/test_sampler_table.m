// Actual GPU sampling beyond Metal's 16 direct sampler arguments, using the production table.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <assert.h>
#include "../runtime/nvn/sampler_table.h"

int main(void) { @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice(); assert(device);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:2 height:1 mipmapped:NO];
    td.storageMode = MTLStorageModeShared; td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [device newTextureWithDescriptor:td];
    const uint8_t pixels[] = {255,0,0,255, 0,255,0,255};
    [texture replaceRegion:MTLRegionMake2D(0,0,2,1) mipmapLevel:0 withBytes:pixels bytesPerRow:8];
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sd.supportArgumentBuffers = YES;
    sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
    id<MTLSamplerState> clamp = [device newSamplerStateWithDescriptor:sd];
    sd.sAddressMode = MTLSamplerAddressModeRepeat;
    id<MTLSamplerState> repeat = [device newSamplerStateWithDescriptor:sd];
    assert(clamp && repeat);
    SamplerTables tables = {0};
    for (unsigned count = 17; count <= 31; count += 14) {
        NSString *source = [NSString stringWithFormat:
            @"#include <metal_stdlib>\nusing namespace metal;\n"
             "struct SamplerTable { array<sampler,32> slots [[id(0)]]; };\n"
             "kernel void main0(constant SamplerTable &table [[buffer(16)]], "
             "texture2d<float> t [[texture(0)]], device float4 *out [[buffer(0)]]) { "
             "sampler s0=table.slots[0]; sampler last=table.slots[%u]; "
             "out[0]=t.sample(s0,float2(1.25,0.5),level(0.0)); "
             "out[1]=t.sample(last,float2(1.25,0.5),level(0.0)); }", count-1];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
        assert(library);
        id<MTLFunction> function = [library newFunctionWithName:@"main0"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        assert(pipeline);
        __unsafe_unretained id<MTLSamplerState> samplers[32];
        for (unsigned i=0;i<count;i++) samplers[i]=clamp;
        samplers[count-1]=repeat;
        id<MTLBuffer> table = sampler_table_buffer(&tables,device,samplers,count); assert(table);
        assert(table == sampler_table_buffer(&tables,device,samplers,count));
        assert(table.length == [function newArgumentEncoderWithBufferIndex:16].encodedLength);
        samplers[0]=repeat;
        assert(table != sampler_table_buffer(&tables,device,samplers,count));
        id<MTLBuffer> output = [device newBufferWithLength:32 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline]; [enc setTexture:texture atIndex:0];
        [enc setBuffer:output offset:0 atIndex:0]; [enc setBuffer:table offset:0 atIndex:16];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc endEncoding]; [cb commit];
        sampler_tables_reset(&tables); // table and its associated sampler lifetimes survive cache teardown
        [cb waitUntilCompleted]; assert(cb.status==MTLCommandBufferStatusCompleted);
        const float *p=output.contents;
        assert(p[0]==0 && p[1]==1 && p[2]==0 && p[3]==1);
        assert(p[4]==1 && p[5]==0 && p[6]==0 && p[7]==1);
    }
    puts("17/31 sampler slots: GPU pixels, distinct addressing, cache reuse/change and teardown passed");
} return 0; }
