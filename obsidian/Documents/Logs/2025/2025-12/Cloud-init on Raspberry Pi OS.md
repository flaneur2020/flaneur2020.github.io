## How can I use it?

- 有这三个文件：
	- meta-data
	- network-config
	- user-data
- meta-data 一般可以不关注

## General configuration (`user-data`)

- 除了网络，基本所有的配置都在 user-data 里面
- 可以用来创建默认 user、定义 locale、安装 package、配置 ssh 等等
- `#cloud-config` 的 header 是强制的

## Networking configuration (`network-config`)

也包括了 netplan

## More about Netplan

- netplan 是 Raspberry Pi OS 中关于网络配置的 primary source of truth
- netplan 的好处是可移植性，可以在不同的 linux distribution 中使用，不管是 NetworkManager 还是 networkd 都可以用
- 直接使用 netplan，可以将配置放在 `/etc/netplan/`
- cloud-init 也会将生成的 network-config 放在里面
- netplan apply 之后，可以将配置应用到 NetworkManager 之类