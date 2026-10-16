// Real translated LDG/STG instructions must declare their access mode for every shader stage/dialect.
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../runtime/nvn/maxwell/shader.h"

int main(void) {
    const int stages[]={STAGE_VERTEX,STAGE_FRAGMENT,STAGE_COMPUTE};
    for(unsigned glsl=0;glsl<2;glsl++) for(unsigned st=0;st<3;st++)
    for(unsigned size=0;size<8;size++) for(unsigned store=0;store<2;store++) {
        _Alignas(uint64_t) uint8_t blob[0xc0]={0};
        unsigned stage=stages[st];
        uint64_t *code=(uint64_t *)(blob+(stage==STAGE_COMPUTE?0:0x80));
        code[1]=(store?0xeed8000000000000ull:0xeed0000000000000ull)|((uint64_t)size<<48)|(7ull<<16)|(2ull<<8)|1;
        code[2]=0xe30000000007000full;  // EXIT
        code[3]=0x50b0000000070f00ull;  // NOP
        ShaderKey key={.glsl=glsl};
        ShaderInfo *info=maxwell_translate(blob,stage==STAGE_COMPUTE?32:0xa0,stage,&key);
        if(!info->msl)fprintf(stderr,"stage %u size %u store %u: %s\n",stage,size,store,info->error);
        assert(info->msl && info->uses_gmem && info->writes_gmem==(bool)store);
        maxwell_free(info);
    }
    puts("96 translated LDG/STG cases: all widths, vertex/fragment/compute, MSL/GLSL access metadata passed");
}
