// Translate real TEXS instructions and compile the resulting Metal, including the 16-sampler boundary.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <assert.h>
#include <string.h>
#include "../runtime/nvn/maxwell/shader.h"

static void instruction(uint64_t *code, unsigned *at, uint64_t word) {
    if (*at % 4 == 0) code[(*at)++] = 0; // Maxwell scheduling word
    code[(*at)++] = word;
}

int main(void) { @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice(); assert(device);
    const int stages[] = {STAGE_VERTEX, STAGE_FRAGMENT, STAGE_COMPUTE};
    const unsigned counts[] = {16,17,31};
    for (unsigned st=0;st<3;st++) for (unsigned n=0;n<3;n++) for (unsigned occupied=0;occupied<3;occupied++) {
        _Alignas(uint64_t) uint8_t blob[0x400]={0};
        int stage=stages[st]; unsigned count=counts[n],at=0;
        if (stage==STAGE_FRAGMENT) *(uint32_t *)(blob+0x78)=15;
        uint64_t *code=(uint64_t *)(blob+(stage==STAGE_COMPUTE?0:0x80));
        if (occupied==1) instruction(code,&at,0xef90000000000000ull|(16ull<<36)|(7ull<<16)|(255ull<<8)|1);
        if (occupied==2) for (unsigned slot=0;slot<=16;slot++)
            instruction(code,&at,0xef90000000000000ull|((uint64_t)slot<<36)|(7ull<<16)|(255ull<<8)|1);
        for (unsigned i=0;i<count;i++)
            instruction(code,&at,0xd000000000000000ull|(1ull<<59)|(2ull<<53)|((8ull+2*i)<<36)|
                (255ull<<28)|(255ull<<20)|(7ull<<16)|(255ull<<8));
        instruction(code,&at,0xe30000000007000full);
        ShaderKey key={0};
        ShaderInfo *info=maxwell_translate(blob,(stage==STAGE_COMPUTE?0:0x80)+at*8,stage,&key);
        if (occupied==2 && count>16) {
            assert(!info->msl && strstr(info->error,"no unused constant-buffer slot"));
            maxwell_free(info);
            continue;
        }
        if (!info->msl) fprintf(stderr,"translate: %s\n",info->error);
        assert(info->msl && info->ntex==(int)count && info->uses_sampler_table==(count>16));
        if (count>16) {
            assert(info->sampler_table_index==(occupied?15:16));
            assert(!(info->cbuf_mask&(1u<<info->sampler_table_index)));
            assert(!strstr(info->msl,"[[sampler(16)]]"));
        }
        NSError *error=nil;
        id<MTLLibrary> library=[device newLibraryWithSource:@(info->msl) options:nil error:&error];
        if (!library) fprintf(stderr,"stage %d count %u: %s\n",stage,count,error.localizedDescription.UTF8String);
        assert(library && [library newFunctionWithName:@"main0"]);
        maxwell_free(info);
    }
    puts("21 TEXS shaders compiled: 16/17/31 samplers and all stages; 6 exhausted-buffer cases rejected cleanly");
} return 0; }
