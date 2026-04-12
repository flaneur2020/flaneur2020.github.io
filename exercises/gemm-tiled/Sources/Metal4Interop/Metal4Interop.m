#import "Metal4Interop.h"

API_AVAILABLE(macos(26.0))
@implementation Metal4Interop

+ (BOOL)commitCommandBuffer:(id<MTL4CommandBuffer>)commandBuffer
                    onQueue:(id<MTL4CommandQueue>)queue
               gpuStartTime:(double *)gpuStartTime
                 gpuEndTime:(double *)gpuEndTime
                      error:(NSError * _Nullable * _Nullable)error
{
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    __block NSError *commitError = nil;
    __block double startTime = 0.0;
    __block double endTime = 0.0;

    MTL4CommitOptions *options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        commitError = feedback.error;
        startTime = feedback.GPUStartTime;
        endTime = feedback.GPUEndTime;
        dispatch_semaphore_signal(semaphore);
    }];

    id<MTL4CommandBuffer> commandBuffers[] = { commandBuffer };
    [queue commit:commandBuffers count:1 options:options];

    dispatch_time_t timeout = dispatch_time(DISPATCH_TIME_NOW, 60ull * NSEC_PER_SEC);
    long waitResult = dispatch_semaphore_wait(semaphore, timeout);
    if (waitResult != 0) {
        if (error != NULL) {
            *error = [NSError errorWithDomain:@"Metal4Interop"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Timed out waiting for MTL4 command buffer completion feedback"}];
        }
        return NO;
    }

    if (gpuStartTime != NULL) {
        *gpuStartTime = startTime;
    }
    if (gpuEndTime != NULL) {
        *gpuEndTime = endTime;
    }
    if (error != NULL) {
        *error = commitError;
    }
    return commitError == nil;
}

@end
