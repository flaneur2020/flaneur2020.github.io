TLDR

- pg 好像做了一个 publish/subscribe 模型，在服务端创建 publication，在客户端创建 subscription 即可；
- 好像使用这个东西不用关注 binlog 的解码这层，能在一开始的时候，将表的 snapshot 传过去，再把行的变更传过去；
- 好像不管 DDL，上下游要 schema 一致；

---

>  the subscriber initially receives a copy of the replicated database object from the publisher and pulls any subsequent changes on the same object as they occur in real-time.

## The architecture

logical replication 遵循一个 publish 和 subscribe 模型。

在 publisher 节点上，会创建一个 publication，用于跟踪一个 table 或者一组 table 的变更。

在 subscriber 节点上，会创建一个 subscription，可以订阅一个或者多个 publication。

replication 的开始，是拷贝当前 publish 的 database 的 snapshot 给 subscriber。这也称作 **table synchronization phase**。

这时可以开多个 worker 来加速这个过程。不过一张表只能有一个 synchronization worker。

在 copy 结束后，publisher 节点后续的 change 会实时地发给 subscriber 节点。

这些 change 在 subscriber 侧会按照 commit 顺序来应用，满足 transactional consistency。

## Basic syntax

```sql
CREATE PUBLICATION my_pub FOR ALL TABLES;
```

```sql
CREATE SUBSCRIPTION my_sub CONNECTION '... <connection string> ...' PUBLICATION my_pub;
```

When the command is run, a logical replication worker is spawned and it receives the logical changes from the publisher. On the publisher side, a _walsender_ process is spawned, which is responsible for reading the WAL one-by-one, decode the changes, and send those changes to the respective subscriber.

Some important points should be noted before using logical replication in versions prior to PostgreSQL 12:

1. Each subscriber can subscribe to multiple publications, and each publication can publish changes to multiple subscribers.
2. To add or remove tables from an existing publication, the [ALTER PUBLICATION](https://www.postgresql.org/docs/10/sql-alterpublication.html) command can be used.
3. The database schema and <mark>DDL definitions cannot be replicated to the subscriber yet. The published tables must exist on the subscriber</mark>. 
4. The replicated table has to be a regular table — not views, materialized views, partition root tables, or foreign tables.
5. <mark>The table should have the same full qualified name in publisher and subscriber</mark>.
6. <mark>The column names must match, but the order of the columns in the subscriber table doesn't matter</mark>. Additionally, there can be the same or more number of columns in a subscribed table.
7. Replication of sequence data and large objects are not yet supported.
