---
layout: post
title: "jedis 代码笔记: JedisPool"
---

最近想了解一下不同语言的连接池的实现，redis 协议是里面最简单的，也借着 jedis 的代码来学习一下 java 里的设计。

## Jedis API

```java
JedisPool jedisPool = new JedisPool(jedisPoolConfig, redisHost, redisPort, timeout, redisPassword);
Jedis jedis = null;
try {
    jedis = jedisPool.getResource();
    jedis.set("key", "val")
} catch (Exception e) {
    logger.error(e.getMessage(), e);
} finally {
    if (jedis != null) {
        jedis.close();
    }
}
```

其中 JedisPool 是一切的入口，随后是从连接池拿连接、执行操作、返还连接三个操作。

## JedisPool
jedis 直接使用 apache common-pool2 的 GenericObjectPool 来管理连接池。

JedisPool 对象继承自 JedisPoolAbstract，然后 JedisPoolAbstract 继承自 Pool<Jedis>，然后 Pool<Jedis> 中包含 protected 成员 GenericObjectPool<T> internalPool。

```
JedisPool
extends JedisPoolAbstract
        extends Pool<Jedis>
              { protected GenericObjectPool<T> internalPool
```

Pool 的构造函数取 final GenericObjectPoolConfig poolConfig 和 PooledObjectFactory<T> factory 两个参数，大致上直接透传给了 GenericObjectPool<T>，poolConfig 用于指定 maxIdle、maxAlive 等连接池常见参数，而 PooledObjectFactory<T> 这里的实现来自 JedisFactory，用于生产 Jedis 对象，每个 Jedis 对象对应一个连接。

GenericObjectPool<T> 中常用的方法大致上有 borrowObject、returnObject、addObject、invalidateObject，对应对象的借用、返还、创建、废弃。

jedisPool.getResource() 会调用 borrowObject()，可能抛出 NoSuchElementException，会转换成 JedisExhaustedPoolException。

## Jedis 对象

Jedis 类继承自 BinaryJedis，实现了很多 Commands 类的接口，似乎通过这些不同的接口对 Redis 的命令做了一定分类：

```
public class Jedis extends BinaryJedis implements JedisCommands, MultiKeyCommands,
    AdvancedJedisCommands, ScriptingCommands, BasicCommands, ClusterCommands, SentinelCommands, ModuleCommands {

  protected JedisPoolAbstract dataSource = null;
```

Jedis 与 BinaryJedis 都是两个又宽又 shallow 的类，里面的方法大约都长同一个样，比如同一个 ping 命令在 Jedis 类中的定义：

```java
public String ping(final String message) {
  checkIsInMultiOrPipeline();
  client.ping(message);
  return client.getBulkReply();
}
```

在 BinaryJedis 类中的定义:

```java
public byte[] ping(final byte[] message) {
  checkIsInMultiOrPipeline();
  client.ping(message);
  return client.getBinaryBulkReply();
}
```

唯一的不同就是 Jedis 类中定义参数、返回值是 String，而 BinaryJedis 中的参数、返回值都是 byte[]。然后 Jedis 类继承了所有 BinaryJedis 的方法，也可以按 byte[] 来操作所有命令。

与 Redis 交互的 Client 定义自 BinaryJedis，Jedis 类通过继承得到。

```java
public class BinaryJedis implements BasicCommands, BinaryJedisCommands, MultiKeyBinaryCommands,
    AdvancedBinaryJedisCommands, BinaryScriptingCommands, Closeable {
  protected Client client = null;
  protected Transaction transaction = null;
  protected Pipeline pipeline = null;
  private final byte[][] dummyArray = new byte[0][];

// ...
```

然后 Client 又是继承自 BinaryClient，两个类也是同样的又 shallow 又宽：

```
public class Client extends BinaryClient implements Commands {
```

然后  BinaryClient 继承自 Connection 最终管理 Redis 连接，终于遇到一个不宽的类了。

```
public class BinaryClient extends Connection {
// ...
  public void ping(final byte[] message) {
    sendCommand(PING, message);
  }
```

Client 与 BinaryClient 之间的关系大致上与 Jedis 与 BinaryJedis 之间的关系相似。那么 Jedis 与 Client 之间有何关系呢？大致上是 Client 对象中的 redis 写命令都是 void 没有返回值，在 Jedis / BinaryJedis 上层会做一点包装，而 Jedis 中的方法一是增加一个 checkIsInMultiOrPipeline() 判断，二是调用类似 client.getBinaryBulkReply() 的方法拿到返回值，三大概就是 Jedis / BinaryJedis 属于 BasicCommands、MultiKeyCommands 等接口的实现，而 Client / BinaryClient 本身并不关注这些接口。个人感觉如果 Client / BinaryClient 类能窄一些，只提供 sendCommand、getBulkReply 等少数几个接口可能会优雅一点。

Jedis、BinaryJedis、Client、BinaryClient、Connection 之间的关系大致上是：

```
Jedis
extends BinaryJedis
      { protected Client client
                  extends BinaryClient
                          extends Connection
```

## Connection

终于走到连接管理的部分了，可见 Connection 类大致是对 socket 的一个包装：

```

public class Connection implements Closeable {

  private static final byte[][] EMPTY_ARGS = new byte[0][];

  private JedisSocketFactory jedisSocketFactory;
  private Socket socket;
  private RedisOutputStream outputStream;
  private RedisInputStream inputStream;
  private boolean broken = false;

```

其中 JedisSocketFactory 会根据对象池的配置来创建 Socket 连接。

RedisOutputStream 和 RedisInputStream 大致上相当于 go 的 bufio，读写都先放到一个固定宽度的 buf 里缓冲一下。

Connection 内部的方法大致上皆为对 Redis 协议的请求 / 响应的简单包装，比如：

```
public void sendCommand(final ProtocolCommand cmd, final byte[]... args);

public void connect();

public String getBulkReply();

public byte[] getBinaryBulkReply();

public Long getIntegerReply();

public List<String> getMultiBulkReply();

public List<byte[]> getBinaryMultiBulkReply();

protected Object readProtocolWithCheckingBroken();

public List<Object> getMany(final int count);

// ...
```

先看 sendCommand 方法：

```java
public void sendCommand(final ProtocolCommand cmd, final byte[]... args) {
  try {
    connect();
    Protocol.sendCommand(outputStream, cmd, args);
  } catch (JedisConnectionException ex) {
    /*
     * When client send request which formed by invalid protocol, Redis send back error message
     * before close connection. We try to read it to provide reason of failure.
     */
    try {
      String errorMessage = Protocol.readErrorLineIfPossible(inputStream);
      if (errorMessage != null && errorMessage.length() > 0) {
        ex = new JedisConnectionException(errorMessage, ex.getCause());
      }
    } catch (Exception e) {
      /*
       * Catch any IOException or JedisConnectionException occurred from InputStream#read and just
       * ignore. This approach is safe because reading error message is optional and connection
       * will eventually be closed.
       */
    }
    // Any other exceptions related to connection?
    broken = true;
    throw ex;
  }
}
```

首先可以看出连接的新建是惰性的，默认连接池中新增连接对象时并非立即创建连接，而是第一个 sendCommand 时尝试 connect。

```java
public void connect() {
  if (!isConnected()) {
    try {
      socket = jedisSocketFactory.createSocket();

      outputStream = new RedisOutputStream(socket.getOutputStream());
      inputStream = new RedisInputStream(socket.getInputStream());
    } catch (IOException ex) {
      broken = true;
      throw new JedisConnectionException("Failed connecting to "
          + jedisSocketFactory.getDescription(), ex);
    }
  }
}
```


判断是否处于连接状态的依据是 socket 字段是否为 null 以及 socket 是否 .isClosed() 等等：

```
public boolean isConnected() {
  return socket != null && socket.isBound() && !socket.isClosed() && socket.isConnected()
      && !socket.isInputShutdown() && !socket.isOutputShutdown();
}
```

更细节的 Redis 协议的读写似乎都集中在 Protocol 这个类里，它的成员方法都是静态方法，直接收 inputStream 或者 outputStream 做参数来操作读或者写。

connect()、sendCommand()、getBulkReply() 乃至 flush() 等操作都有可能抛出异常 JedisConnectionException，JedisConnectionException 属于不可恢复的错误，如果遇到该异常则将该 connection 标记为 broken。

此外有一个小细节是，因为 outputStream 有 buffer 缓冲，因此在所有读操作中皆执行一发 flush()：

```
public byte[] getBinaryBulkReply() {
  flush();
  return (byte[]) readProtocolWithCheckingBroken();
}
```

## Conclusion
* jedis 直接使用 apache-commons2 的 GenericObjectPool<T> 作为连接池，大大简化了连接池相关的管理工作，apache-commons2 似乎是个宝藏；
* 每个连接在 connect()、读、写操作中都有可能遇到 IOException，会给包成 JedisConnectionException，一旦出现，则视为不可恢复性异常，因为有连接池的存在，单个连接对象不需要考虑重试；
* 读取响应结果的方法前面先 flush() 一发写入，这个细节似乎满需要注意的；
* Jedis / BinaryJedis 与 Client / BinaryClient 四个类又 shallow 又宽，BinaryClient  继承自 Connection 类而不是组合 Connection 对象，感觉这两个地方略不优雅，猜可能也与 Pipeline、Transaction、Cluster 等高级 Redis 操作的接口有关，后面详细看一下；

