https://polaris.io/#tag/Overview
https://other-docs.snowflake.com/en/polaris/overview

![[Pasted image 20240924122037.png]]
好像说过去 iceberg 是一个单表的表格式，catalog 是一个 restful 规范，可以把 N 张表挂在一个 catalog 的 namespace 中。

## Key concepts

![[Pasted image 20240924122136.png]]
namespace 还有层级关系。

## Catalog

可以创建多个 catalog 来管理 iceberg tables。

catalog 需要满足支持原子地更新 metadata pointer 到新版本 table 的能力。

catalog 分为 internal 和 external：

1. internal：在 polaris catalog 中管理的 table
2. external：在其他 iceberg catalog provider 中管理的 table，比如 snowflake、Glue、Dremio Arctic，这些 catalog 中的表会被 sync 到 polaris catalog 中；这些 table 在 polaris catalog 中是 readonly 的；

## Namespace

一个 catalog 可以有多个 namesapce；namespace 可以嵌套；

## Service Principal

在 catalog 中创建的一个实体，每个 service principal 封装特定的 credential 在里面。

query engine 使用 service principal 来访问 catalogs。

polaris catalog 会为每个 service principal 生成一个 client id 和 client secret。

## Service connection

每个 service connection 表示一个能够读写 polaris catalog 的 REST-compatible engine（比如 flink、spark、trino）。

当你新建一个 service connection，polaris catalog admin 讲 service principal grant 给一个特定的 principal role 给新的 service connection。

## Storage configuration

> A storage configuration stores a generated identity and access management (IAM) entity for your external cloud storage and is created when you create a catalog. The storage configuration is used to set the values to connect Polaris Catalog to your cloud storage. During the catalog creation process, an IAM entity is generated and used to create a trust relationship between the cloud storage provider and Polaris Catalog.