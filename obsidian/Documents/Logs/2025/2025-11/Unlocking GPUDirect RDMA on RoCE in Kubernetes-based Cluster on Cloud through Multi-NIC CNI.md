
## GPUDirect RDMA on Kubernetes Cluster on Cloud

- GPUDirect RDMA
- nvidia 的 network operator，配合 GPU operator，能够无痛地在 k8s 集群中协同 GPUDirect RDMA；
- network operator 会使用 multus；这里有两个关键的 resource： NicClusterPolicy 和 HostDeviceNetwork；
- NicClusterPolicy 定义了网络设备的 Cluster 状态；
- HostDeviceNetwork 定用于创建 Network Attachment Definition 资源；生成的 NAD 会被 multus 消费，从而将第二张网卡挂在打过注解的 Pod 里面。

![[Pasted image 20251125134931.png]]

## RoCE and Its Challenges on Virtual Private Cloud

- RoCE 使用 IP 地址作为最初的连接建立，在 RDMA 数据传输中就不用了。
- 和 TCP/IP 不通，RoCE 需要在 Pod 中完整地拥有 host device；
- 这时 host-interface-local 的 IPAM 就不是必要了，直接 assign host 的 IP 作为 Pod IP 更合适；
- 然而，在传统 CNI 框架下，将 Host IP 作为 Pod IP 有两个挑战；
- 必须使用静态 IPAM 为每台主机上的每个设备指定特定的 IP 地址，并且 Pod 必须通过注解来得到这部分信息；
- 常见的 device plugin 往往只支持基于数量的设备分配，而且分配的顺序不能确定，必须和 Pod IP 对应的设备对齐；

![[Pasted image 20251125135623.png]]
 

## **Multi-NIC CNI for GPUDirect RDMA on RoCE**

- v1.2.0 开始，Multi-NIC CNI 支持了 GPUDirect RDMA on RoCE.
- 只需要定义 MultiNicNetwork 资源，即可在私有云 VPC 环境中自动化 RoCE 主机设备，不需要手工配置设备定义、Pod 注解，也不需要解决设备分配顺序问题；

```
apiVersion: multinic.fms.io/v1  
kind: MultiNicNetwork  
metadata:  
name: multinic-mellanox-hostdevice  
spec:  
ipam: |  
{  
"type": "host-device-ipam"  
}  
multiNICIPAM: false  
plugin:  
cniVersion: "0.3.1"  
type: mellanox
```

- Multi-NIC CNI 对 RoCE 上 GPUDirect RDMA 的支持依赖两个 CNI 组件：`mellanox` 和 `host-device-ipam`。
- **`mellanox` 插件**：作为主设备配置插件，Multi-NIC CNI 运算符会动态生成一个对应的 `HostDeviceNetwork` 定义，用于与 NVIDIA 网络运算符的状态同步，并创建一个多网卡的 Network Attachment Definition（NAD）资源。当 Multus 消费该 NAD 时，流程如下：首先调用 Multi-NIC CNI，再由其委托给 `host-device` CNI 执行实际配置。
- **`host-device-ipam` 插件**：作为 IPAM（IP 地址管理）插件，Multi-NIC CNI 内置逻辑可识别主机设备的 IP 地址，并**严格按照设备插件分配的顺序**，为每个设备生成带有静态 IPAM 配置的 `host-device` CNI 配置，完成委托流程。