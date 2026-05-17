USRBIO 是 3fs 的高性能 IO 卢金。用户可以通过 USRBIO 提交 IO 请求到 FUSE 进程，绕开 FUSE 的限制，比如 FUSE 限制每个 IO 操作的 size，对网络文件系统会特别不友好。

## 概念

- **iov**：一大块共享的内存，用于 zero-copy 的读写操作，用户 fuse 进程两者共享，能够通过 FUSE 进程进行 IB 内存注册。
- **ior**：一个共享内存的 ring，用于用户进程和 FUSE 进程之间通信；使用起来类似 linux 的 io_uring，IO 会按 batch 进行执行；对多线程程序来讲，推荐使用多个 ring 来减少同步开销；
- **File descriptor Registration**: 只有注册过的 fd 可以用于 USRBIO。

## 函数

### hf3fs_iorcreate4 / hf3fs_iordestroy

创建 ior。

```c
int hf3fs_iorcreate4(struct hf3fs_ior *ior,
                     const char *hf3fs_mount_point,
                     int entries,
                     bool for_read,
                     int io_depth,
                     int timeout,
                     int numa,
                     uint64_t flags);
```

其中 hf3fs_mount_point 对应 3fs 在本地的 mount 路径。

```c
void hf3fs_destroy(struct hf3fs_ior *ior);
```

### hf3fs_iovcreate / hf3fs_iovdestroy

```c
int hf3fs_iovcreate(struct hf3fs_iov *iov,
                    const char *hf3fs_mount_point,
                    size_t size,
                    size_t block_size,
                    int numa);
```

```c
void hf3fs_iovdestroy(struct hf3fs_iov *iov);
```

### hf3fs_reg_fd / hf3fs_dereg_fd

```c
int hf3fs_reg_fd(int fd, uint64_t flags);
```

注册一个 file descriptor 用于 IO。这里的 fd 应该需要是 3fs 的一个文件的 fd。

```c
int fd = open("example.txt", O_RDONLY);
hf3fs_reg_fd(fd, 0);
hf3fs_dereg_fd(fd);
close(fd);
```

### hf3fs_prep_io

```c
int hf3fs_prep_io(struct hf3fs_ior *ior,
                  const struct hf3fs_iov *iov,
                  bool read,
                  void *ptr,
                  int fd,
                  size_t off,
                  uint64_t len,
                  void *userdata);
```

如果提交成功，会将 io 操作记录到 ior 中。

其中的 userdata 可以说任意数据，可以在 hf3fs_wait_for_ios 中返回回来。

### hf3fs_submit_ios

```c
int hf3fs_submit_ios(const struct hf3fs_ior *ior);
```

可以幂等地触发。

### hf3fs_wait_for_ios

```c
int hf3fs_wait_for_ios(const struct hf3fs_ior *ior,
                       struct hf3fs_cqe *cqes,
                       int cqec,
                       int min_results,
                       const struct timespec *abs_timeout);
```

`cqes` 是 `hf3fs_cqe` 的地址，会包含 IO 操作的结果，并带着 userdata 回来。