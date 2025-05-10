> Large scale machine learning workloads on [Kubernetes](https://kubernetes.io/) commonly suffer from the lack of a resource reservation system to allow gang scheduling. For example, a training job requiring 100 GPU pods can be stuck with only 80 out of 100 GPUs provisioned, while other pods are stuck in a pending state. This is common when there are other jobs competing for resources in the same K8s cluster.

> MCAD allows you to queue each of your Ray workloads until resource availability requirements are met. With MCAD, your Ray cluster’s pods will only be created once there is a guarantee that all of the pods can be scheduled.

## Workload requirements

在使用 Ray 做大规模的训练时，有这两种能力的支持会有很大的收益：

1. Gang Scheduling：只有当所有的本地资源都到位的时候，才启动整个 Ray cluster；
2. Workload 抢占：在有高优先级的 workload 时，抢占低优先级的 Ray workload；

其中 #2 的抢占，是 MACD 的未来的计划。#1 的 Gang scheduling 是 MACD 已经支持的。

值得一提的是，Ray 内部也有 gang-scheduling 支持，叫做 Placement groups，它允许在 Ray 集群内部的资源都到位之后，才执行这个 ray 的 task。

## Components for scaling workloads on Kubernetes

### Multi-Cluster-App-Dispatcher (MCAD)

MACD 能够管理一个或者多个 k8s 集群之间的 batch jobs。

![[Pasted image 20250501201704.png]]

- MACD 提供了一个 AppWrapper 能够包裹用户的 workload 比如 Deployment、Job、PodGroup 等在里面
- User objects within an `AppWrapper` are queued until aggregated resources are available in one of the Kubernetes clusters.

## Running the workload

```yaml
kind: AppWrapper
metadata:
  name: raycluster-glue
spec:
  resources:
    custompodresources:
      - replicas: 2
        requests:
          cpu: "3"
          memory: "16G"
          nvidia.com/gpu: "1"
        limits:
          cpu: "3"
          memory: "16G"
          nvidia.com/gpu: "1"
    generictemplate:
      kind: RayCluster
      spec:
        headGroupSpec:
          containers:
            - image: projectcodeflare/codeflare-glue:latest
              resources:
                limits:
                  cpu: "2"
                  memory: "16G"
                  nvidia.com/gpu: "0"
                requests:
                  cpu: "2"
                  memory: "16G"
                  nvidia.com/gpu: "0"
        workerGroupSpecs:
          - replicas: 1
            containers:
              - image: projectcodeflare/codeflare-glue:latest
                resources:
                  limits:
                    cpu: "4"
                    memory: "16G"
                    nvidia.com/gpu: "2"
                  requests:
                    cpu: "4"
                    memory: "16G"
                    nvidia.com/gpu: "2"

```