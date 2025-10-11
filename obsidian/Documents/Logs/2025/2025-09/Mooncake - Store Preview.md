
- 内置 Transfer Engine，能够拿掉冗余的内存复制，multi-NIC GPUDirect RDMA pooling 来最大限度利用网络传输
- 支持 striping、并发 IO 传输；
- 可以做分层缓存；

## Architecture

![[Pasted image 20250907130633.png]]

有两个组件：

1. Master Service：管理一个逻辑上的内存池；还有一个 metadata service（etcd、redis 等）需要 transfer engine 来访问，需要单独部署；
2. Client：既是访问 cache 的客户端，也是管理 cache 块的 storage segment server。

```protobuf
message BufHandle {
  required uint64 segment_name = 1;  // Storage segment name (can be simply understood as the name of the storage node)
  required uint64 size = 2;          // Size of the allocated space
  required uint64 buffer = 3;        // Pointer to the allocated space

  enum BufStatus {
    INIT = 0;          // Initial state, space reserved but not used
    COMPLETE = 1;      // Completed usage, space contains valid data
    FAILED = 2;        // Usage failed, upstream should update the handle state to this value
    UNREGISTERED = 3;  // Space has been unregistered, metadata deleted
  }
  required BufStatus status = 4 [default = INIT]; // Space status
};

message ReplicaInfo {
  repeated BufHandle handles = 1; // Specific locations of the stored object data

  enum ReplicaStatus {
    UNDEFINED = 0;   // Uninitialized
    INITIALIZED = 1; // Space allocated, waiting for write
    PROCESSING = 2;  // Writing data in progress
    COMPLETE = 3;    // Write completed, replica available
    REMOVED = 4;     // Replica has been removed
    FAILED = 5;      // Replica write failed, consider reallocation
  }
  required ReplicaStatus status = 2 [default = UNDEFINED]; // Replica status
};

service MasterService {
  // Get the list of replicas for an object
  rpc GetReplicaList(GetReplicaListRequest) returns (GetReplicaListResponse);

  // Get replica lists for objects matching a regex
  rpc GetReplicaListByRegex(GetReplicaListByRegexRequest) returns (GetReplicaListByRegexResponse);

  // Start Put operation, allocate storage space
  rpc PutStart(PutStartRequest) returns (PutStartResponse);

  // End Put operation, mark object write completion
  rpc PutEnd(PutEndRequest) returns (PutEndResponse);

  // Delete all replicas of an object
  rpc Remove(RemoveRequest) returns (RemoveResponse);

  // Remove objects matching a regex
  rpc RemoveByRegex(RemoveByRegexRequest) returns (RemoveByRegexResponse);

  // Storage node (Client) registers a storage segment
  rpc MountSegment(MountSegmentRequest) returns (MountSegmentResponse);

  // Storage node (Client) unregisters a storage segment
  rpc UnmountSegment(UnmountSegmentRequest) returns (UnmountSegmentResponse);
}
```


