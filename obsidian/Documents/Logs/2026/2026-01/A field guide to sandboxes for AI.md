![[Pasted image 20260111152012.png]]

## The three-question model

- Boundary：限定 isolation 生效的位置
	- container：进程在不同的 namespace、仍然共享同一个 kernel
	- gVisor：workload 的 syscall 限定在一个 userspace kernel 上
	- MicroVM：syscall 打到 guest kernel 上；host kernel 可以看到 hypervisor、VMM 的活动，看不到 guest kernel 的 ABI；
	- runtime boundary：guest 没有 syscall ABI；只能调用显式暴露的 API；
- Policy 是 boundary 内部代码可以做哪些事情：
	- filesystem read/write/exec
	- 网络访问目标、协议
	- 进程的创建、信号
	- 设备访问（比如 GPU）
	- 时间、内存、CPU、Disk Quota
	- syscall、ioctl、imports 等等
- Lifecycle：执行期间的持久性
	- Fresh Run：什么都不保留，执行完就完全销毁
	- Workspace：长生命周期的 filesystem、session；如果 secret 被泄露、persistence 被滥用，会比较危险；
	- Snapshot/restore: 可以通过 checkpointing 快速恢复；对于 RL rollout 和 "prewarm" agent 会比较友好；

### The three questions

考虑一个 sandbox 方案时，需要考虑：

1. what's shared between the code and the host?
2. what can the code touch (files, network, devices, syscalls)
3. what survives between runs?

## Linux building blocks

- namespace
- capabilities: 将 root 的权限拆细，container 一般都是通过一个精简过的 capability set；`CAP_SYS_ADMIN` 不大好，相当于还是 root 的所有权限；
- cgroups
- seccomp
	- seccomp 是 syscall filtering；在 syscall entry 中插入 eBPF 实现；这个程序可以决定 allow, deny, log, trap, kill 或者通知报警；
	- 通过限制系统调用（syscalls）来最小化内核攻击面，防止权限提升（privilege escalation，比如通过 `ptrace`, `mount`, `kexec_load`, `bpf`, `perf_event_open`, `userfaultfd`, etc），并淘汰不安全的遗留接口
	- 在真实沙箱中，seccomp 不仅**过滤系统调用（syscall）本身**，还会**检查参数**（如 `clone3` 的标志、`ioctl` 的请求码），并采用 **白名单（allowlist）机制**（只允许已知安全的调用）
	- seccomp 还支持 **用户通知机制（SECCOMP_RET_USER_NOTIF）**：当进程发起系统调用时，内核会**暂停它**，并将请求转发给一个**用户态的监督进程（broker）**，由其根据策略动态决定是否允许（例如：只允许打开 `/tmp/` 下的文件，或通过网络代理过滤连接），这个灵活性会很高

### How containers combine these

1. Namespaces scope/virtualize resources.
2. **Capabilities are reduced.**
3. Cgroups cap resource usage.
4. **Seccomp filters syscalls on entry.**
5. A root filesystem provides the container’s view of `/` (often layered via overlayfs).
6. **AppArmor/SELinux may apply additional policy.**

container 有了 policy-based restriction，但是仍然是 shared kernel boundary。You reduce what the process can see (namespaces), cap what it can consume (cgroups), and restrict which syscalls it can invoke (seccomp). 但是 boundary 并没有更强。

## Where containers fail

作者表示 container 并非一个安全的 hostile code 的 boundary。

除了 misconfiguration 和 kernel/runtime bugs，在 AI 时代还有第三个风险：policy leakage；

### Misconfiguration escapes

大多数容器逃逸并非漏洞所致，而往往是人为主动削弱了隔离。比如：

1. `--preivileged` 移除了几乎所有的安全机制
2. mount 了 `/var/run/docker.sock` 到容器里
3. 可写的 `/sys` 或 `/proc/sys`：可以修改内核参数、关闭内核的保护，可以绕过 seccomp、审计等等
4. 可写的宿主机路径（bind-mount）：可以打通 file system namespace，修改宿主机的关键文件
5. 赋予高危的 capability： CAP_SYS_ADMIN
6. 加入宿主机的 namespace：`--pid=host`、`--net=host`
7. 设备直通：可以绕过内核，直接通过驱动的 bug 获取内存访问

### Kernel and runtime bugs

一个配置正确的 container 仍然访问着同一个 host kernel。

如果宿主机 kernel 有了安全漏洞，那么所有的容器仍然不安全。比如：

- **Dirty COW** ([CVE-2016-5195](https://nvd.nist.gov/vuln/detail/CVE-2016-5195)): copy-on-write race in the memory subsystem.
- **Dirty Pipe** ([CVE-2022-0847](https://www.docker.com/blog/vulnerability-alert-avoiding-dirty-pipe-cve-2022-0847-on-docker-engine-and-docker-desktop/)): pipe handling bug enabling overwriting data in read-only mappings.
- **fs_context overflow** ([CVE-2022-0185](https://nvd.nist.gov/vuln/detail/CVE-2022-0185)): filesystem context parsing bug exploited in container contexts.

Seccomp reduces exposure by blocking syscalls, but the syscalls you allow are still kernel code. Docker’s default seccomp profile is a compatibility-biased allowlist: it blocks some high-risk calls but still permits hundreds.

而且这甚至不需要 kernel 一个 container runtime 的 bug 就足够了(比如 runC overwrite: [CVE-2019-5736](https://nvd.nist.gov/vuln/detail/CVE-2019-5736)).

### Policy leakage (the AI-specific one)

很多时候安全的威胁甚至不是容器逃逸，而是策略配置错误。

- 如果 Agent **能读你的代码仓库** → 它可以把代码发到外网。
- 如果 Agent **能读 `~/.aws/credentials`** → 它能偷走你的云账号。
- 如果 Agent **能访问内部服务（如数据库、CI/CD）** → 它能横向移动，攻陷整个内网。

这不是逃逸，而是合法的滥用。

在这个层面上，只依赖 boundary 就不够，还需要 policy。

## Choosing a sandbox

Before picking a boundary, I also write down a **minimum viable policy**. If you can’t enforce these, you don’t have a sandbox yet:

- **Default-deny outbound network**, then allowlist. (Or route everything through a policy proxy.)
- **No long-lived credentials** in the sandbox. Use short-lived scoped tokens.
- **Workspace-only filesystem** access. No host mounts besides what you explicitly intend.
- **Resource limits**: CPU, memory, disk, timeouts, and PIDs.
- **Observability**: log process tree, network egress, and failures. Sandboxes without telemetry become incident-response theater.