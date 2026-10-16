// Small standalone iPad hardware probe. Build products and reports belong outside the source checkout.
#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <mach/mach.h>
#include <os/proc.h>
#include <sys/utsname.h>

static NSDictionary *shader_probe(void) {
    id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();
    NSString *folder = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    NSMutableArray *results = [NSMutableArray new];
    for (NSString *stage in @[@"VS", @"FS"]) {
        NSString *path = [folder stringByAppendingPathComponent:[NSString stringWithFormat:@"failed-%@.metal", stage]];
        NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
        if (!source) continue;
        for (int variant = 0; variant < 3; variant++) {
            @autoreleasepool {
                MTLCompileOptions *options = [MTLCompileOptions new];
                options.preserveInvariance = variant != 2;
                if (variant == 1) options.optimizationLevel = MTLLibraryOptimizationLevelSize;
                NSString *text = variant == 2 ? [source stringByReplacingOccurrencesOfString:@"position, invariant" withString:@"position"] : source;
                NSError *error = nil;
                NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
                fprintf(stderr, "SHADER_PROBE starting %s variant %d\n", stage.UTF8String, variant);
                id<MTLLibrary> library = [gpu newLibraryWithSource:text options:options error:&error];
                NSDictionary *r = @{@"stage":stage, @"variant":@(variant), @"success":@(library != nil),
                    @"seconds":@(NSDate.timeIntervalSinceReferenceDate - start), @"error":error.localizedDescription ?: @""};
                [results addObject:r];
                fprintf(stderr, "SHADER_PROBE %s\n", r.description.UTF8String);
            }
        }
    }
    return @{@"gpu":gpu.name ?: @"", @"shader_tests":results, @"available_memory":@(os_proc_available_memory())};
}

static NSDictionary *probe(void) {
    id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();
    struct utsname machine;
    uname(&machine);
    NSMutableDictionary *r = [@{
        @"model": @(machine.machine),
        @"os": UIDevice.currentDevice.systemVersion,
        @"physical_memory": @(NSProcessInfo.processInfo.physicalMemory),
        @"available_memory": @(os_proc_available_memory()),
        @"gpu": gpu.name ?: @"unavailable",
        @"max_buffer_length": @(gpu.maxBufferLength),
        @"recommended_working_set": @(gpu.recommendedMaxWorkingSetSize),
        @"bc_textures": @(gpu.supportsBCTextureCompression),
        @"float32_filtering": @(gpu.supports32BitFloatFiltering),
        @"metalfx_spatial": @([MTLFXSpatialScalerDescriptor supportsDevice:gpu]),
    } mutableCopy];
    NSMutableArray *maps = [NSMutableArray new];
    for (NSNumber *base in @[@(0x7000000000ull), @(0x400000000ull), @(0x800000000ull)]) {
        vm_address_t address = base.unsignedLongLongValue;
        vm_size_t length = 0x320000000ull;
        kern_return_t kr = vm_allocate(mach_task_self(), &address, length, VM_FLAGS_FIXED);
        [maps addObject:@{@"base": base, @"size": @(length), @"result": @(kr)}];
        if (kr == KERN_SUCCESS) vm_deallocate(mach_task_self(), address, length);
    }
    r[@"guest_reservations"] = maps;
    NSMutableArray *regions = [NSMutableArray new];
    vm_address_t cursor = 0;
    for (int n = 0; n < 100; n++) {
        vm_size_t length = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        kern_return_t kr = vm_region_64(mach_task_self(), &cursor, &length, VM_REGION_BASIC_INFO_64,
            (vm_region_info_t)&info, &count, &object);
        if (object) mach_port_deallocate(mach_task_self(), object);
        if (kr != KERN_SUCCESS || !length) break;
        [regions addObject:@{@"base":@(cursor),@"size":@(length),@"protection":@(info.protection)}];
        cursor += length;
    }
    r[@"mapped_regions"] = regions;
    NSMutableArray *allocations = [NSMutableArray new];
    for (NSNumber *length in @[@(64ull<<20),@(256ull<<20),@(1ull<<30),@(2ull<<30),@(3ull<<30),@(4ull<<30),@(8ull<<30)]) {
        vm_address_t address = 0;
        kern_return_t kr = vm_allocate(mach_task_self(), &address, length.unsignedLongLongValue, VM_FLAGS_ANYWHERE);
        [allocations addObject:@{@"base":@(address),@"size":length,@"result":@(kr)}];
        if (kr == KERN_SUCCESS) vm_deallocate(mach_task_self(), address, length.unsignedLongLongValue);
    }
    r[@"available_reservations"] = allocations;
    vm_address_t compact = 0x7000000000ull;
    kern_return_t compact_result = vm_allocate(mach_task_self(), &compact, 0xA4000000ull, VM_FLAGS_FIXED);
    r[@"compact_fixed_reservation"] = @(compact_result);
    if (compact_result == KERN_SUCCESS) vm_deallocate(mach_task_self(), compact, 0xA4000000ull);
    // A bounded test: never attempt the game's multi-gigabyte GPU buffer on a small-memory device.
    vm_address_t small = 0;
    vm_size_t size = 16u << 20;
    if (vm_allocate(mach_task_self(), &small, size, VM_FLAGS_ANYWHERE) == KERN_SUCCESS) {
        @autoreleasepool {
            id<MTLBuffer> buffer = [gpu newBufferWithBytesNoCopy:(void *)small length:size
                options:MTLResourceStorageModeShared deallocator:nil];
            r[@"zero_copy_16mb"] = @(buffer != nil);
            buffer = nil;
        }
        vm_deallocate(mach_task_self(), small, size);
    }
    r[@"available_memory_after_probe"] = @(os_proc_available_memory());
    return r;
}

@interface ProbeScene : UIResponder <UIWindowSceneDelegate>
@property(strong, nonatomic) UIWindow *window;
@end
@implementation ProbeScene
- (void)scene:(UIScene *)scene willConnectToSession:(UISceneSession *)session options:(UISceneConnectionOptions *)options {
    (void)session; (void)options;
    self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene *)scene];
    UIViewController *view = [UIViewController new];
    UITextView *text = [[UITextView alloc] initWithFrame:self.window.bounds];
    text.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    text.editable = NO;
    text.font = [UIFont monospacedSystemFontOfSize:14 weight:UIFontWeightRegular];
    view.view = text;
    self.window.rootViewController = view;
    [self.window makeKeyAndVisible];
    NSDictionary *result = getenv("LARECOMP_PROBE_SHADER") ? shader_probe() : probe();
    NSData *json = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil];
    NSString *folder = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    [json writeToFile:[folder stringByAppendingPathComponent:@"device-probe.json"] atomically:YES];
    text.text = [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
    fprintf(stderr, "LAS_DEVICE_PROBE %s\n", text.text.UTF8String);
    fflush(stderr);
    if (getenv("LARECOMP_PROBE_EXIT"))
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{ exit(0); });
}
@end

@interface ProbeDelegate : UIResponder <UIApplicationDelegate>
@end
@implementation ProbeDelegate
@end

int main(int argc, char **argv) {
    @autoreleasepool { return UIApplicationMain(argc, argv, nil, NSStringFromClass(ProbeDelegate.class)); }
}
