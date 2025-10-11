> A fast cold start is critical for ensuring your deployment can react quickly to traffic changes without large delays

![[Pasted image 20250903212716.png]]

作者在这篇文章中，将一个 8B 模型的冷启动时间，缩小到了 26s。

## What does an LLM container include?

比如一个 llama 3.1 8B 的模型镜像中，可能存在：

1. CUDA：2.8GB
2. Torch：1.4GB
3. Other Libs：1GB
4. Model Weights：15GB

这使得整个镜像达到了 20GB。

## What do cold and warm starts mean?

warm 是指 k8s 的节点上曾经跑过这个镜像，可以直接 load 镜像启动。

## What happens when starting an LLM container?

```
Deployment Timeline: 
Image Pull:      5–6 min   ███████ 
Layer Extract:   3–4 min   ████ 
Config & Start:   ~2 min   █                  
Total:           ~11 min
```

可见大部分时间都在 prepare container to start 这层。

## Rethink image pulling and model loading

### Step 1: 直接从 object store 下载镜像

直接从 GCS、S3 这样的地方去下载镜像。

|**Method**|**Speed** **(depending on machine/network)**|**Time**|
|---|---|---|
|Cloud Registry Pull (GAR/ECR)|60 MB/s|~350s|
|Internal Registry Pull (Harbor)|120 MB/s|~170s|
|Direct GCS/S3 Download|2 GB/s or higher|~10s|

### Step 2: Skip extraction with FUSE

> Using the seekable-tar format as the foundation, we **separated the model weights from the container image and enabled direct loading into GPU memory.**

相当于将对象存储 mount 到本地，从而使模型镜像能够按需加载到显存。

### Step 3: Load models directly into GPU memory

原本的模式：下载镜像、写到磁盘、装到内存。

作者提供了一个直接将模型参数写入到 GPU 内存的办法。

