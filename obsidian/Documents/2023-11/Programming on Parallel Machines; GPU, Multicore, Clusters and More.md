https://heather.cs.ucdavis.edu/parprocbook


Note carefully that a call to the kernel doesn’t block; it returns immediately. For that reason, the code above has a host barrier call, to avoid copying the results back to the host from the device before they’re ready: 

	cudaThreadSynchronize();

On the other hand, if our code were to have another kernel call, say on the next line after
    
	find1elt<<<dimGrid,dimBlock>>>(dm,drs,n);
    
and if some of the second call’s input arguments were the outputs of the first call, there would be an implied barrier between the two calls; the second would not start execution before the first finished.

Calls like `cudaMemcpy()` do block until the operation completes.

There is also a thread barrier available for the threads themselves, at the block level. The call is

	__syncthreads();


This can only be invoked by threads within a block, not across blocks. In other words, this is barrier synchronization within blocks.


## 5.4 Understanding the Hardware Structure

if a warp of threads needs to access global memory (including local memory; see below), the SM will schedule some other warp while the memory access is pending.


The key implication is that shared memory is used essentially as a programmer-managed cache.

Data will start out in global memory, but if a variable is to be accessed multiple times by the GPU code, it’s probably better for the programmer to write code that copies it to shared memory,