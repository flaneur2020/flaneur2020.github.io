
fizzbee 的语法是 starlark 语言，是 python 的一个子集；正常的表达式可以是一个标准的 python REPL；

## High level structure

- fizz.yaml 用来放配置。
- .fizz 文件是模型描述文件；

### fizz.yaml 文件：

```
options:
  maxActions: 10
  maxConcurrentActions: 2
  
actionOptions:
  YourActionName:
    maxActions: 1
```

### .fizz file

包含如下部分：

```
# init action

# invariants

# action1

# action2

# additional_fuctions 
```
## Actions

actions 是 model 中的主要部分。表示 model 到达一个 state 之前的一系列 steps。

action 的定义：

```
[atomic|serial] action YourActionName
  # Almost python code
  a += 1
  b += 1
```

atomic actions:

在 TLA+ 中，action 是原子的，而在 fizzbee 中它是一个可选配置。

如果配置为 atomic action，在上面的例子中，a 和 b 的递增操作是原子的；

serial action 中的每行语句，会产生一个 yield point。

a 和 b 会按顺序递增；

## Block modifiers

每个 block 可以有一个 block modifier，包括 `atomic`, `serial`, `parallel`, `oneof`.

oneof 写起来大约是： 

```
action IncrementAny:
  oneof:
    a += 1
    b += 1
```

parallel 写起来大约是：

```
action IncrementAny:
  parallel:
    a += 1
    b += 1
```

如果 a=0, b=0，那么后续的路径可能是：

- a=1, b=1
- a=0, b=1
- a=1, b=0

## init action

`Init` 会在 model checking 之前先执行，比如：

```
action Init:
  # More common usecases will use `any` statements
  oneof:
    atomic:
        a = 0
        b = 10
    atomic:
        a = 10
        b = 0
```

## Function

Function 目前还不大完整，不能支持传参数。后面会加。

```
func TossACoin:
  oneof:
    return 0
    return 1
```

## Control Flow

- if else
- while
- for
- any

## Invariants/Assertions

```
always assertion FirstInvariant:
  return a == b
  
always assertion SecondInvariant2:
  # it can have loops, if-else, etc.
  return some_boolean_expression
```

另一个写法是：

```
invariant:
  # Here each statement is a separate invariant.
  always a == 10
  always a < 10
  always b < 10
```

除了 always，还有 always eventually、eventually always。

