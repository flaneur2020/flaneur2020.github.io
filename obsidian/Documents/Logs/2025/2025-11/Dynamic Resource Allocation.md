DRA 主要用于管理向 Pod 插入各种硬件的加速器。

> With DRA, device drivers and cluster admins define device _classes_ that are available to _claim_ in workloads. Kubernetes allocates matching devices to specific claims and places the corresponding Pods on nodes that can access the allocated devices.

申请 DRA 资源的流程和 dynamic volume provisioning 类似，比如你使用 PersistentVolumeClaim 来申明特定的 storage class 的资源，并在 Pod 中引用它。

## Benefits of DRA

| 传统 Device Plugin 的问题 | DRA 的解决方案                                           |
| -------------------- | --------------------------------------------------- |
| 每个容器需单独声明设备数量        | **Pod 级别声明**：通过 `ResourceClaim` 引用，无需在容器中写设备数量      |
| 不支持设备共享              | **多 Pod/容器共享同一设备**（如共享 GPU 显存）                      |
| 无设备筛选能力              | **CEL 表达式过滤**：按属性（颜色、型号、版本等）精细筛选设备                  |
| 无集中设备分类              | **DeviceClass**：定义设备类别（如“高性能”、“低成本”），供用户按需申请        |
| 无生命周期管理              | **ResourceClaimTemplate**：自动生成/绑定/清理资源，与 Pod 生命周期一致 |
|                      |                                                     |

## 例子

```yaml
# ResourceSlice - 由驱动自动创建，管理员无需手动修改
apiVersion: resource.k8s.io/v1
kind: ResourceSlice
metadata:
  name: gpu-slice-node1
spec:
  driver: nvidia-dra-driver
  pool:
    name: nvidia-gpu-pool
    generation: 1
  allNodes: false  # 仅本节点可访问
  nodeName: node-1
  devices:
    - name: gpu-0
      attributes:
        vendor:
          string: "NVIDIA"
        model:
          string: "A100-80GB"
        memory:
          string: "80Gi"
        allowMultipleAllocations: true  # ✅ 支持显存分割共享
      capacity:
        memory:
          value: "80Gi"
          requestPolicy:
            default: "1Gi"
            validRange:
              min: "1Gi"
              step: "1Gi"
    - name: gpu-1
      attributes:
        vendor:
          string: "NVIDIA"
        model:
          string: "A100-40GB"
        memory:
          string: "40Gi"
        allowMultipleAllocations: true
      capacity:
        memory:
          value: "40Gi"
          requestPolicy:
            default: "1Gi"
            validRange:
              min: "1Gi"
              step: "1Gi"

```

### 定义设备类别

```yaml
apiVersion: resource.k8s.io/v1
kind: DeviceClass
metadata:
  name: high-performance-gpu
spec:
  selectors:
    - cel:
        expression: |
          device.attributes["vendor"].string == "NVIDIA" &&
          device.attributes["model"].string == "A100-80GB"
    - cel:
        expression: |
          device.attributes["vendor"].string == "NVIDIA" &&
          device.attributes["model"].string == "A100-40GB"
  extendedResourceName: nvidia.com/gpu  # ✅ 兼容传统 extended resource 用法
```

### 创建 ResourceClaimTemplate

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: ai-training-gpu-template
spec:
  spec:
    devices:
      requests:
        - name: gpu-memory
          firstAvailable:  # ✅ 优先级列表：先尝试 80GB，不行再用 40GB
            - name: a100-80gb
              deviceClassName: high-performance-gpu
              capacity:
                requests:
                  memory: 20Gi  # 每个 Pod 需要 20Gi 显存
            - name: a100-40gb
              deviceClassName: high-performance-gpu
              capacity:
                requests:
                  memory: 20Gi
```

### 部署 Pod

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-training-job
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-trainer
  template:
    metadata:
      labels:
        app: ai-trainer
    spec:
      containers:
        - name: trainer
          image: nvcr.io/nvidia/pytorch:24.01-py3
          command: ["python", "train.py"]
          # ✅ 不需要声明任何资源！设备通过 ResourceClaim 传递
          # 传统方式：resources: { limits: { "nvidia.com/gpu": 1 } } ❌ 不再需要
      # ✅ 引用模板，K8s 自动为每个 Pod 创建独立的 ResourceClaim
      resourceClaimTemplates:
        - metadata:
            name: gpu-memory
          spec:
            name: ai-training-gpu-template
```
