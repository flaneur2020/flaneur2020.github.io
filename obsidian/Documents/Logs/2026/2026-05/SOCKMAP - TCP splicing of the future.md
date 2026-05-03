TLDR：

- TCP splicing 是 proxy 界的圣杯
- SOCKMAP 是 cilium 一个老哥加进来的，eBPF hook
- SOCKMAP 挂载点不是一个 sock，而是一个 map，这个 map 中可以挂多个 socket 的描述符；
- sockmap 能通过 bpf_sk_redirect_map 将__sk_buff 转给别的 socket；
- 但是作者测 echo server 测下来，sockmap 是最慢的，splice() 也没多快，简单的 read/write 反而最快；


![[Pasted image 20260501214213.png]]