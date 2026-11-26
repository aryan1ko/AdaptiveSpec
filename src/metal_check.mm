#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        if (!device) {
            std::cerr << "No Metal device found\n";
            return 1;
        }

        std::cout << "GPU: "
                  << [[device name] UTF8String]
                  << "\n";

        NSString* source = @R"(
            #include <metal_stdlib>
            using namespace metal;

            kernel void test_kernel(
                device float* data [[buffer(0)]],
                uint id [[thread_position_in_grid]]
            ) {
                data[id] *= 2.0f;
            }
        )";

        NSError* error = nil;

        id<MTLLibrary> library =
            [device newLibraryWithSource:source
                    options:nil
                    error:&error];

        if (!library) {
            std::cerr << [[error localizedDescription] UTF8String]
                      << "\n";
            return 1;
        }

        std::cout << "Metal compilation successful\n";
    }

    return 0;
}
