#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

API_AVAILABLE(macos(26.0))
@interface Metal4Interop : NSObject
+ (BOOL)commitCommandBuffer:(id<MTL4CommandBuffer>)commandBuffer
                    onQueue:(id<MTL4CommandQueue>)queue
               gpuStartTime:(double *)gpuStartTime
                 gpuEndTime:(double *)gpuEndTime
                      error:(NSError * _Nullable * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
