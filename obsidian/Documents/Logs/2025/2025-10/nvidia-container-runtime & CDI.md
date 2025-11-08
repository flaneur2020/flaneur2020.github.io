
tldr

- 好像 device-plugin 只解决和调度器的集成问题，真的初始化环境还是靠 nvidia-container-runtime。
- 初始化环境好像就是在容器镜像的基础上，把驱动相关的各种文件挂到容器里，并初始化一些环境变量啥的；
- nvidia-container-runtime 作用在 runc 的 prestart 钩子上；
- （ODI 好像也是做这个？）
- （容器镜像中只需包含 `libodi` 客户端，无需 `libnvidia-container`）
---

https://github.com/cncf-tags/container-device-interface

![[Pasted image 20251023211344.png]]

nvidia software stack:

- libnvidia-container: 将 GPU 向 docker 容器暴露，允许容器访问到 GPU driver，从而允许执行 GPU 应用
- nvidia-container-runtime：确保 docker 容器正确地配置，从而允许访问 GPU，它会包一把 runc，来确保 GPU drivers、libraries、device files 正常地在容器中展现；

**nvidia-container-runtime** 保证在容器启动时，能够访问宿主机的  GPU 资源。

libnvidia-container 是 nvidia-container-runtime 和 nvidia-container-cli 共同依赖的库；

Prestart Hook：会保障所有的 nvidia library、drivers、device files 在容器中被包含在内；

（看起来就是确保 nvidia 相关的所有的包，在容器镜像启动前，先链接进来）

```
[用户提交 Pod] 
        ↓
[Kubernetes Scheduler] → 根据 nvidia.com/gpu: 1 → 选择有 GPU 的节点
        ↓
[kubelet 在节点上准备容器] 
        ↓
[调用 NVIDIA Device Plugin] → 获取分配的 GPU 设备（如 /dev/nvidia0）
        ↓
[调用 nvidia-container-runtime 启动容器]
        ↓
[nvidia-container-runtime 自动注入]
   - /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm
   - NVIDIA 驱动库（/usr/lib/x86_64-linux-gnu/libcuda.so）
   - 环境变量 NVIDIA_VISIBLE_DEVICES=0
        ↓
[容器内运行 nvidia-smi / CUDA 程序] → ✅ 成功！
```

使用 odi 之后：

```yaml
resources:
  limits:
    device.odinvidia.com/gpu: 1   # ← 注意这个资源名！
```